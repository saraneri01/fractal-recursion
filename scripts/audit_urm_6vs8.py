"""Diagnostic comparison of URM-F (six applications) and the eight-application pilot on 30 training-pipeline batches (paper, Appendix B).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import fmean
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import train.urm as pt
ROOT = Path(__file__).resolve().parents[1]
CONFIG_A = ROOT / 'configs/urm/fractal_seed0_60ep.yaml'
CKPT_A = ROOT / 'checkpoints/urm/fractal_seed0_60ep/step_2405.pt'
CONFIG_F = ROOT / 'configs/urm/fractal_8calls_seed0_30ep.yaml'
CKPT_F = ROOT / 'checkpoints/urm/fractal_8calls_seed0_30ep/step_2405_uncompiled_audit_v1.pt'
DATA = ROOT / 'datasets/urm/arc-aug-1000-eval-v1'
OUT = ROOT / 'runs/urm_diagnostics/urmf_6vs8_step2405_30batches_v1.json'
AUDIT_BATCH_SIZE = 2
N_BATCHES = 30
if OUT.exists():
    raise SystemExit(f'Refusing to overwrite existing audit: {OUT}')
OUT.parent.mkdir(parents=True, exist_ok=True)

def load_config(path: Path, checkpoint: Path):
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    config = pt.PretrainConfig(**raw)
    config.data_path = str(DATA)
    config.load_checkpoint = str(checkpoint)
    config.load_optimizer_state = False
    config.ema = False
    config.global_batch_size = AUDIT_BATCH_SIZE
    config.loop_deltas = [0]
    return config

def unwrap_model(model):
    """
    Return the actual URM/URM-F model used for the dynamics audit.

    Handles:
      1. torch.compile (_orig_mod)
      2. distributed/module wrappers
      3. ACTLossHead -> underlying recurrent model

    The returned object must expose both initial_carry() and .inner.
    """
    while True:
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
            continue
        if hasattr(model, 'module'):
            model = model.module
            continue
        if not hasattr(model, 'inner') and hasattr(model, 'model') and hasattr(model.model, 'inner'):
            model = model.model
            continue
        break
    if not hasattr(model, 'inner'):
        raise RuntimeError(f'Could not unwrap to URM model; final type={type(model)}')
    if not hasattr(model, 'initial_carry'):
        raise RuntimeError(f'Unwrapped model lacks initial_carry(); type={type(model)}')
    return model

def to_device_batch(raw_batch, device):
    """
    Extract the actual tensor dictionary from the official URM dataloader.

    Observed loader format:
        ["all", {"inputs": ..., "labels": ..., "puzzle_identifiers": ...}, batch_size]

    Also accepts a bare dictionary for robustness.
    """
    if isinstance(raw_batch, dict):
        batch = raw_batch
    elif isinstance(raw_batch, (tuple, list)):
        dict_items = [x for x in raw_batch if isinstance(x, dict)]
        if len(dict_items) != 1:
            raise TypeError(f'Expected exactly one dict inside batch wrapper, found {len(dict_items)}; type={type(raw_batch)}')
        batch = dict_items[0]
    else:
        raise TypeError(f'Unexpected batch type: {type(raw_batch)}')
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

def rms(x: torch.Tensor) -> float:
    x = x.detach().float()
    return float(torch.sqrt(torch.mean(x * x)).item())

def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(a.shape[0], -1)
    b = b.detach().float().reshape(b.shape[0], -1)
    return float(F.cosine_similarity(a, b, dim=1, eps=1e-08).mean().item())

def diagnostic_values(pre: torch.Tensor, post: torch.Tensor, input_embeddings: torch.Tensor, injection: torch.Tensor):
    delta = post - pre
    pre_rms = rms(pre)
    post_rms = rms(post)
    delta_rms = rms(delta)
    return {'pre_rms': pre_rms, 'post_rms': post_rms, 'delta_rms': delta_rms, 'relative_delta': delta_rms / max(pre_rms, 1e-12), 'cos_pre_post': cosine_flat(pre, post), 'input_rms': rms(input_embeddings), 'injection_rms': rms(injection), 'cos_delta_input': cosine_flat(delta, input_embeddings), 'cos_post_input': cosine_flat(post, input_embeddings)}

def apply_shared_core(inner, hidden_states, injection, seq_info):
    """
    Exactly the common operation used by both implementations:
        hidden <- hidden + injection
        hidden <- same registered URM layers
    """
    hidden_states = hidden_states + injection
    for layer in inner.layers:
        hidden_states = layer(hidden_states=hidden_states, **seq_info)
    return hidden_states

def make_initial_inner_carry(model, batch):
    """
    Build the official initial carry on the same device as the model.

    The URM implementation creates some carry tensors with bare torch
    constructors, so when called outside the original training context they
    otherwise default to CPU. For this audit we reproduce the intended CUDA
    device context explicitly.
    """
    device = next(model.parameters()).device
    with torch.device(device):
        carry = model.initial_carry(batch)
    return model.inner.reset_carry(carry.halted, carry)

def valid_ce(logits, labels):
    """
    Diagnostic gradient objective ONLY.
    This is not reported as the official URM training loss.
    We only use it to inspect relative gradient propagation through
    the six final-H-cycle core applications.
    """
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.long().reshape(-1), ignore_index=-100)

def audit_baseline(model, batch, batch_index):
    inner = model.inner
    inner_carry = make_initial_inner_carry(model, batch)
    seq_info = dict(cos_sin=inner.rotary_emb())
    input_embeddings = inner._input_embeddings(batch['inputs'], batch['puzzle_identifiers'])
    h = inner_carry.current_hidden
    with torch.no_grad():
        for _ in range(inner.config.H_cycles - 1):
            for _ in range(inner.config.L_cycles):
                h = apply_shared_core(inner, h, input_embeddings, seq_info)
    h = h.detach().requires_grad_(True)
    records = []
    tracked_states = []
    for call_idx in range(1, inner.config.L_cycles + 1):
        pre = h
        injection = input_embeddings
        h = apply_shared_core(inner, pre, injection, seq_info)
        h.retain_grad()
        tracked_states.append(h)
        rec = diagnostic_values(pre, h, input_embeddings, injection)
        rec.update({'batch': batch_index, 'model': 'URM', 'call': call_idx, 'label': f'A{call_idx}', 'state_updated': 'H', 'direct_input': True, 'zM_zL_cosine': None})
        records.append(rec)
    logits = inner.lm_head(h)[:, inner.puzzle_emb_len:]
    loss = valid_ce(logits, batch['labels'])
    model.zero_grad(set_to_none=True)
    loss.backward()
    for rec, state in zip(records, tracked_states):
        rec['grad_rms'] = rms(state.grad) if state.grad is not None else None
    return (records, float(loss.detach().item()))

def audit_fractal(model, batch, batch_index):
    inner = model.inner
    core_calls = inner.config.M_cycles * (inner.config.L_cycles_inner + 1)
    model_label = f'URM-F-{core_calls}'
    inner_carry = make_initial_inner_carry(model, batch)
    seq_info = dict(cos_sin=inner.rotary_emb())
    input_embeddings = inner._input_embeddings(batch['inputs'], batch['puzzle_identifiers'])
    z_M = inner_carry.current_hidden
    z_L = inner_carry.current_hidden_L
    with torch.no_grad():
        for _ in range(inner.config.H_cycles - 1):
            for _ in range(inner.config.M_cycles):
                for _ in range(inner.config.L_cycles_inner):
                    z_L = apply_shared_core(inner, z_L, z_M + input_embeddings, seq_info)
                z_M = apply_shared_core(inner, z_M, z_L, seq_info)
    z_M = z_M.detach().requires_grad_(True)
    z_L = z_L.detach().requires_grad_(True)
    records = []
    tracked_states = []
    call_idx = 0
    for m_idx in range(1, inner.config.M_cycles + 1):
        for l_idx in range(1, inner.config.L_cycles_inner + 1):
            call_idx += 1
            pre = z_L
            injection = z_M + input_embeddings
            z_L = apply_shared_core(inner, pre, injection, seq_info)
            z_L.retain_grad()
            tracked_states.append(z_L)
            rec = diagnostic_values(pre, z_L, input_embeddings, injection)
            rec.update({'batch': batch_index, 'model': model_label, 'call': call_idx, 'label': f'M{m_idx}.L{l_idx}', 'state_updated': 'L', 'direct_input': True, 'zM_zL_cosine': cosine_flat(z_M, z_L)})
            records.append(rec)
        call_idx += 1
        pre = z_M
        injection = z_L
        z_M = apply_shared_core(inner, pre, injection, seq_info)
        z_M.retain_grad()
        tracked_states.append(z_M)
        rec = diagnostic_values(pre, z_M, input_embeddings, injection)
        rec.update({'batch': batch_index, 'model': model_label, 'call': call_idx, 'label': f'M{m_idx}.M', 'state_updated': 'M', 'direct_input': False, 'zM_zL_cosine': cosine_flat(z_M, z_L)})
        records.append(rec)
    expected_calls = inner.config.M_cycles * (inner.config.L_cycles_inner + 1)
    assert call_idx == expected_calls, f'Observed {call_idx} fractal core calls, expected {expected_calls}'
    logits = inner.lm_head(z_M)[:, inner.puzzle_emb_len:]
    loss = valid_ce(logits, batch['labels'])
    model.zero_grad(set_to_none=True)
    loss.backward()
    for rec, state in zip(records, tracked_states):
        rec['grad_rms'] = rms(state.grad) if state.grad is not None else None
    return (records, float(loss.detach().item()))

def summarize(records):
    groups = {}
    for r in records:
        key = (r['model'], r['call'], r['label'], r['state_updated'])
        groups.setdefault(key, []).append(r)
    summary = []
    numeric_keys = ['pre_rms', 'post_rms', 'delta_rms', 'relative_delta', 'cos_pre_post', 'input_rms', 'injection_rms', 'cos_delta_input', 'cos_post_input', 'grad_rms', 'zM_zL_cosine']
    for (model, call, label, state_updated), rows in sorted(groups.items()):
        item = {'model': model, 'call': call, 'label': label, 'state_updated': state_updated, 'direct_input': rows[0]['direct_input'], 'n_batches': len(rows)}
        for k in numeric_keys:
            vals = [x[k] for x in rows if x.get(k) is not None and (not (isinstance(x.get(k), float) and math.isnan(x[k])))]
            item[k] = fmean(vals) if vals else None
        summary.append(item)
    return summary

def print_summary(summary):
    print()
    print('===== INTERNAL DYNAMICS SUMMARY =====')
    print(f'{'MODEL':<7} {'CALL':<7} {'STATE':<5} {'DIRECT_X':<8} {'REL_DELTA':>10} {'COS_PREPOST':>12} {'GRAD_RMS':>12} {'COS_DX':>10} {'COS_ML':>10}')
    for r in summary:
        ml = r['zM_zL_cosine']
        ml_s = 'NA' if ml is None else f'{ml:.5f}'
        gr = r['grad_rms']
        gr_s = 'NA' if gr is None else f'{gr:.3e}'
        print(f'{r['model']:<7} {r['label']:<7} {r['state_updated']:<5} {str(r['direct_input']):<8} {r['relative_delta']:>10.5f} {r['cos_pre_post']:>12.5f} {gr_s:>12} {r['cos_delta_input']:>10.5f} {ml_s:>10}')

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this audit.')
    print('===== URM / URM-F INTERNAL DYNAMICS AUDIT =====')
    print('DATA SPLIT: TRAIN/DEVELOPMENT ONLY')
    print('FINAL TEST IS NOT ACCESSED')
    print('batch size:', AUDIT_BATCH_SIZE)
    print('batches:', N_BATCHES)
    print()
    config_A = load_config(CONFIG_A, CKPT_A)
    config_F = load_config(CONFIG_F, CKPT_F)
    train_loader, train_metadata_A = pt.create_dataloader(config_A, 'train', test_set_mode=False, epochs_per_iter=1, global_batch_size=AUDIT_BATCH_SIZE, rank=0, world_size=1)
    _, train_metadata_F = pt.create_dataloader(config_F, 'train', test_set_mode=False, epochs_per_iter=1, global_batch_size=AUDIT_BATCH_SIZE, rank=0, world_size=1)
    torch.manual_seed(config_A.seed)
    state_A = pt.init_train_state(config_A, train_metadata_A, rank=0, world_size=1)
    torch.manual_seed(config_F.seed)
    state_F = pt.init_train_state(config_F, train_metadata_F, rank=0, world_size=1)
    model_A = unwrap_model(state_A.model)
    model_F = unwrap_model(state_F.model)
    model_A.eval()
    model_F.eval()
    device_A = next(model_A.parameters()).device
    device_F = next(model_F.parameters()).device
    if device_A != device_F:
        raise RuntimeError(f'Models on different devices: A={device_A}, F={device_F}')
    print('A checkpoint step:', state_A.step)
    print('F checkpoint step:', state_F.step)
    print('device:', device_A)
    pA = sum((p.numel() for p in model_A.parameters() if p.requires_grad))
    pF = sum((p.numel() for p in model_F.parameters() if p.requires_grad))
    print('A parameters:', pA)
    print('F parameters:', pF)
    assert pA == pF
    calls_A = model_A.inner.config.M_cycles * (model_A.inner.config.L_cycles_inner + 1)
    calls_F = model_F.inner.config.M_cycles * (model_F.inner.config.L_cycles_inner + 1)
    print('Left model core calls/H-cycle:', calls_A)
    print('Right model core calls/H-cycle:', calls_F)
    assert calls_A == 6
    assert calls_F == 8
    records = []
    losses_A = []
    losses_F = []
    loader_it = iter(train_loader)
    for batch_idx in range(1, N_BATCHES + 1):
        raw_batch = next(loader_it)
        batch = to_device_batch(raw_batch, device_A)
        needed = {'inputs', 'labels', 'puzzle_identifiers'}
        missing = needed - set(batch)
        if missing:
            raise KeyError(f'Missing batch keys {missing}; available={sorted(batch)}')
        rec_A, loss_A = audit_fractal(model_A, batch, batch_idx)
        rec_F, loss_F = audit_fractal(model_F, batch, batch_idx)
        records.extend(rec_A)
        records.extend(rec_F)
        losses_A.append(loss_A)
        losses_F.append(loss_F)
        print(f'batch {batch_idx}/{N_BATCHES}: diagnostic_CE A={loss_A:.6f} F={loss_F:.6f}')
        torch.cuda.empty_cache()
    summary = summarize(records)
    print_summary(summary)
    result = {'purpose': 'Post-hoc TRAIN/development diagnostic of internal dynamics. No final-test data used. Diagnostic CE is used only to inspect gradient propagation and is not the official training metric.', 'baseline_config': str(CONFIG_A), 'baseline_checkpoint': str(CKPT_A), 'fractal_config': str(CONFIG_F), 'fractal_checkpoint': str(CKPT_F), 'data': str(DATA), 'split': 'train', 'audit_batch_size': AUDIT_BATCH_SIZE, 'n_batches': N_BATCHES, 'parameter_count_A': pA, 'parameter_count_F': pF, 'diagnostic_ce_A_mean': fmean(losses_A), 'diagnostic_ce_F_mean': fmean(losses_F), 'summary': summary, 'raw_records': records}
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print()
    print('Saved:', OUT)
    print('AUDIT COMPLETE')
if __name__ == '__main__':
    main()

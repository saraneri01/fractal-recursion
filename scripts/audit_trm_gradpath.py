"""Gradient-path audit for TRM and TRM-F: both must perform 21 core applications with gradients through the final seven.
"""
from __future__ import annotations
import dataclasses
import gc
import json
import math
from pathlib import Path
import torch
import yaml
from models.trm_baseline import TinyRecursiveReasoningModel_ACTV1 as TRM
from models.trm_fractal import TinyRecursiveReasoningModel_ACTV1 as TRMF
ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / 'configs/trm/baseline_seed0_train.yaml'
DATA_META = ROOT / 'datasets/trm/arc1concept-aug-1000/train/dataset.json'
OUT = ROOT / 'runs/audits/audit_trm_gradpath.json'
DEVICE = torch.device('cuda')
BATCH = 1
SEED = 12345
if OUT.exists():
    raise SystemExit(f'Refusing to overwrite {OUT}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA required')
with open(BASE_CONFIG) as f:
    saved_cfg = yaml.safe_load(f)
with open(DATA_META) as f:
    meta = json.load(f)
base_cfg = dict(saved_cfg['arch'])
base_cfg.pop('name', None)
base_cfg.pop('loss', None)
base_cfg.update(batch_size=BATCH, seq_len=meta['seq_len'], vocab_size=meta['vocab_size'], num_puzzle_identifiers=meta['num_puzzle_identifiers'])
fractal_cfg = dict(base_cfg)
fractal_cfg.update(M_cycles=2, L_cycles_inner=2)

def move_carry(carry, device):
    values = {}
    for field in dataclasses.fields(carry):
        x = getattr(carry, field.name)
        values[field.name] = x.to(device)
    return type(carry)(**values)

def run_backward_audit(model_cls, cfg, label):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = model_cls(cfg).to(DEVICE)
    model.train()
    model.zero_grad(set_to_none=True)
    call_info = []
    received_gradients = {}

    def forward_hook(_module, _inputs, output):
        idx = len(call_info)
        info = {'call_index': idx, 'grad_enabled': bool(torch.is_grad_enabled()), 'output_requires_grad': bool(output.requires_grad)}
        call_info.append(info)
        if output.requires_grad:

            def backward_hook(grad, call_idx=idx):
                norm = float(grad.float().norm().item())
                received_gradients[call_idx] = {'norm': norm, 'finite': bool(torch.isfinite(grad).all().item()), 'nonzero': bool(norm > 0.0)}
                return grad
            output.register_hook(backward_hook)
    handle = model.inner.L_level.register_forward_hook(forward_hook)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(20261007)
    inputs = torch.randint(low=1, high=meta['vocab_size'], size=(BATCH, meta['seq_len']), dtype=torch.int32, device=DEVICE, generator=g)
    puzzle_identifiers = torch.randint(low=1, high=meta['num_puzzle_identifiers'], size=(BATCH,), dtype=torch.int32, device=DEVICE, generator=g)
    batch = {'inputs': inputs, 'puzzle_identifiers': puzzle_identifiers}
    carry = model.inner.empty_carry(BATCH)
    carry = move_carry(carry, DEVICE)
    reset = torch.ones(BATCH, dtype=torch.bool, device=DEVICE)
    carry = model.inner.reset_carry(reset, carry)
    new_carry, logits, _q = model.inner(carry, batch)
    loss = logits.float().square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f'{label}: non-finite audit loss')
    loss.backward()
    handle.remove()
    core_param_grads = {}
    present = 0
    finite = 0
    nonzero = 0
    squared_total = 0.0
    for name, p in model.inner.L_level.named_parameters():
        if p.grad is None:
            core_param_grads[name] = {'present': False, 'finite': False, 'norm': None, 'nonzero': False}
            continue
        present += 1
        grad_f = p.grad.float()
        is_finite = bool(torch.isfinite(grad_f).all().item())
        norm = float(grad_f.norm().item())
        is_nonzero = norm > 0.0
        finite += int(is_finite)
        nonzero += int(is_nonzero)
        squared_total += norm * norm
        core_param_grads[name] = {'present': True, 'finite': is_finite, 'norm': norm, 'nonzero': is_nonzero}
    n_core_params = len(core_param_grads)
    grad_enabled_indices = [x['call_index'] for x in call_info if x['grad_enabled']]
    requires_grad_indices = [x['call_index'] for x in call_info if x['output_requires_grad']]
    backward_indices = sorted(received_gradients.keys())
    result = {'label': label, 'loss': float(loss.item()), 'num_core_calls': len(call_info), 'grad_enabled_indices': grad_enabled_indices, 'output_requires_grad_indices': requires_grad_indices, 'backward_received_indices': backward_indices, 'per_call_backward': {str(k): v for k, v in sorted(received_gradients.items())}, 'core_parameter_tensors': n_core_params, 'core_parameter_grads_present': present, 'core_parameter_grads_finite': finite, 'core_parameter_grads_nonzero': nonzero, 'core_total_grad_norm': math.sqrt(squared_total), 'core_parameter_details': core_param_grads}
    del loss, logits, new_carry, carry, batch, inputs
    del puzzle_identifiers, model
    gc.collect()
    torch.cuda.empty_cache()
    return result
print('===== BACKWARD AUDIT: BASELINE TRM =====')
A = run_backward_audit(TRM, base_cfg, 'TRM')
print('calls:', A['num_core_calls'], '| grad-enabled:', A['grad_enabled_indices'], '| backward:', A['backward_received_indices'])
print('core grad tensors:', f'{A['core_parameter_grads_present']}/{A['core_parameter_tensors']}', '| nonzero:', f'{A['core_parameter_grads_nonzero']}/{A['core_parameter_tensors']}', '| total grad norm:', A['core_total_grad_norm'])
print()
print('===== BACKWARD AUDIT: TRM-F =====')
F = run_backward_audit(TRMF, fractal_cfg, 'TRM-F')
print('calls:', F['num_core_calls'], '| grad-enabled:', F['grad_enabled_indices'], '| backward:', F['backward_received_indices'])
print('core grad tensors:', f'{F['core_parameter_grads_present']}/{F['core_parameter_tensors']}', '| nonzero:', f'{F['core_parameter_grads_nonzero']}/{F['core_parameter_tensors']}', '| total grad norm:', F['core_total_grad_norm'])
expected_final_calls = list(range(14, 21))
checks = {'baseline_21_calls': A['num_core_calls'] == 21, 'fractal_21_calls': F['num_core_calls'] == 21, 'baseline_grad_enabled_final_7': A['grad_enabled_indices'] == expected_final_calls, 'fractal_grad_enabled_final_7': F['grad_enabled_indices'] == expected_final_calls, 'baseline_outputs_require_grad_final_7': A['output_requires_grad_indices'] == expected_final_calls, 'fractal_outputs_require_grad_final_7': F['output_requires_grad_indices'] == expected_final_calls, 'baseline_backward_reaches_all_final_7': A['backward_received_indices'] == expected_final_calls, 'fractal_backward_reaches_all_final_7': F['backward_received_indices'] == expected_final_calls, 'baseline_final_call_grads_finite_nonzero': all((A['per_call_backward'][str(i)]['finite'] and A['per_call_backward'][str(i)]['nonzero'] for i in expected_final_calls)), 'fractal_final_call_grads_finite_nonzero': all((F['per_call_backward'][str(i)]['finite'] and F['per_call_backward'][str(i)]['nonzero'] for i in expected_final_calls)), 'baseline_core_parameters_receive_grad': A['core_parameter_grads_present'] == A['core_parameter_tensors'], 'fractal_core_parameters_receive_grad': F['core_parameter_grads_present'] == F['core_parameter_tensors'], 'baseline_core_total_grad_nonzero': A['core_total_grad_norm'] > 0.0, 'fractal_core_total_grad_nonzero': F['core_total_grad_norm'] > 0.0}
result = {'purpose': 'Verify that TRM-F does not reproduce the HRM-C1 credit-assignment failure: both architectures must expose the same final 7/21 shared-core calls to backward.', 'baseline': A, 'fractal': F, 'checks': checks}
OUT.write_text(json.dumps(result, indent=2))
print()
print('===== FORMAL BACKWARD CHECKS =====')
for key, value in checks.items():
    print(f'{key}: {('PASS' if value else 'FAIL')}')
overall = all(checks.values())
print()
print('OVERALL:', 'PASS' if overall else 'FAIL')
print('saved:', OUT)
if not overall:
    raise SystemExit('BACKWARD GRADIENT AUDIT FAILED')

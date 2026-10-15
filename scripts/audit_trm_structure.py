"""Structural audit of TRM and TRM-F: parameter counts, parameter and buffer signatures and state-dict keys.
"""
from __future__ import annotations
import dataclasses
import gc
import json
from pathlib import Path
import torch
import yaml
from models.trm_baseline import TinyRecursiveReasoningModel_ACTV1 as TRM
from models.trm_fractal import TinyRecursiveReasoningModel_ACTV1 as TRMF
ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / 'configs/trm/baseline_seed0_train.yaml'
DATA_META = ROOT / 'datasets/trm/arc1concept-aug-1000/train/dataset.json'
OUT = ROOT / 'runs/audits/audit_trm_structure.json'
AUDIT_BATCH = 1
DEVICE = torch.device('cuda')
if OUT.exists():
    raise SystemExit(f'Refusing to overwrite: {OUT}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA required for forward audit.')
with open(BASE_CONFIG) as f:
    full_cfg = yaml.safe_load(f)
with open(DATA_META) as f:
    meta = json.load(f)
arch = dict(full_cfg['arch'])
arch.pop('name', None)
arch.pop('loss', None)
arch.update(batch_size=AUDIT_BATCH, seq_len=meta['seq_len'], vocab_size=meta['vocab_size'], num_puzzle_identifiers=meta['num_puzzle_identifiers'])
arch_f = dict(arch)
arch_f.update(M_cycles=2, L_cycles_inner=2)

def tensor_signature(named_items):
    return {name: {'shape': list(t.shape), 'dtype': str(t.dtype)} for name, t in named_items}

def structure_signature(model):
    params = tensor_signature(model.named_parameters())
    buffers = tensor_signature(model.named_buffers())
    return {'trainable_parameters': sum((p.numel() for p in model.parameters() if p.requires_grad)), 'all_parameters': sum((p.numel() for p in model.parameters())), 'parameter_signature': params, 'buffer_signature': buffers, 'state_dict_keys': list(model.state_dict().keys())}

def move_carry(carry, device):
    values = {}
    for f in dataclasses.fields(carry):
        x = getattr(carry, f.name)
        values[f.name] = x.to(device)
    return type(carry)(**values)

def audit_model(model_cls, config, label):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = model_cls(config)
    sig = structure_signature(model)
    model = model.to(DEVICE)
    model.train()
    call_grad_flags = []

    def hook(_module, _args):
        call_grad_flags.append(bool(torch.is_grad_enabled()))
    handle = model.inner.L_level.register_forward_pre_hook(hook)
    B = AUDIT_BATCH
    batch = {'inputs': torch.zeros((B, meta['seq_len']), dtype=torch.int32, device=DEVICE), 'puzzle_identifiers': torch.zeros((B,), dtype=torch.int32, device=DEVICE)}
    carry = model.inner.empty_carry(B)
    carry = move_carry(carry, DEVICE)
    reset = torch.ones(B, dtype=torch.bool, device=DEVICE)
    carry = model.inner.reset_carry(reset, carry)
    with torch.enable_grad():
        new_carry, logits, q = model.inner(carry, batch)
    handle.remove()
    result = {'label': label, 'structure': sig, 'core_calls_total': len(call_grad_flags), 'core_calls_with_grad': sum(call_grad_flags), 'core_calls_no_grad': len(call_grad_flags) - sum(call_grad_flags), 'grad_pattern': call_grad_flags, 'logits_shape': list(logits.shape), 'logits_requires_grad': bool(logits.requires_grad), 'q_halt_shape': list(q[0].shape), 'q_continue_shape': list(q[1].shape)}
    del new_carry, logits, q, carry, batch, model
    gc.collect()
    torch.cuda.empty_cache()
    return result
print('===== CONFIG USED =====')
print('H_cycles:', arch['H_cycles'])
print('baseline L_cycles:', arch['L_cycles'])
print('TRM-F M_cycles:', arch_f['M_cycles'])
print('TRM-F L_cycles_inner:', arch_f['L_cycles_inner'])
print('L_layers:', arch['L_layers'])
print('hidden_size:', arch['hidden_size'])
print('num_heads:', arch['num_heads'])
print('seq_len:', arch['seq_len'])
print('vocab_size:', arch['vocab_size'])
print('num_puzzle_identifiers:', arch['num_puzzle_identifiers'])
print()
print('===== BASELINE TRM =====')
A = audit_model(TRM, arch, 'TRM')
print('params:', A['structure']['all_parameters'], '| core calls:', A['core_calls_total'], '| grad calls:', A['core_calls_with_grad'])
print()
print('===== FRACTAL TRM-F =====')
F = audit_model(TRMF, arch_f, 'TRM-F')
print('params:', F['structure']['all_parameters'], '| core calls:', F['core_calls_total'], '| grad calls:', F['core_calls_with_grad'])
same_param_count = A['structure']['all_parameters'] == F['structure']['all_parameters']
same_trainable_count = A['structure']['trainable_parameters'] == F['structure']['trainable_parameters']
same_param_signature = A['structure']['parameter_signature'] == F['structure']['parameter_signature']
same_buffer_signature = A['structure']['buffer_signature'] == F['structure']['buffer_signature']
same_state_keys = A['structure']['state_dict_keys'] == F['structure']['state_dict_keys']
expected_A_calls = arch['H_cycles'] * (arch['L_cycles'] + 1)
expected_F_calls = arch_f['H_cycles'] * (arch_f['M_cycles'] * (arch_f['L_cycles_inner'] + 1) + 1)
expected_A_grad = arch['L_cycles'] + 1
expected_F_grad = arch_f['M_cycles'] * (arch_f['L_cycles_inner'] + 1) + 1
core_calls_pass = A['core_calls_total'] == expected_A_calls == F['core_calls_total'] == expected_F_calls == 21
grad_calls_pass = A['core_calls_with_grad'] == expected_A_grad == F['core_calls_with_grad'] == expected_F_grad == 7
no_grad_calls_pass = A['core_calls_no_grad'] == 14 and F['core_calls_no_grad'] == 14
output_shape_pass = A['logits_shape'] == F['logits_shape'] and A['q_halt_shape'] == F['q_halt_shape'] and (A['q_continue_shape'] == F['q_continue_shape'])
results = {'baseline': A, 'fractal': F, 'checks': {'parameter_count_parity': same_param_count, 'trainable_parameter_count_parity': same_trainable_count, 'parameter_signature_parity': same_param_signature, 'persistent_buffer_signature_parity': same_buffer_signature, 'state_dict_key_parity': same_state_keys, 'core_calls_21_vs_21': core_calls_pass, 'grad_calls_7_vs_7': grad_calls_pass, 'no_grad_calls_14_vs_14': no_grad_calls_pass, 'output_shape_parity': output_shape_pass}}
OUT.write_text(json.dumps(results, indent=2))
print()
print('===== FORMAL AUDIT =====')
for k, v in results['checks'].items():
    print(f'{k}: {('PASS' if v else 'FAIL')}')
print()
print('TRM calls pattern:')
print(A['grad_pattern'])
print()
print('TRM-F calls pattern:')
print(F['grad_pattern'])
all_pass = all(results['checks'].values())
print()
print('OVERALL:', 'PASS' if all_pass else 'FAIL')
print('saved:', OUT)
if not all_pass:
    raise SystemExit('STRUCTURAL AUDIT FAILED')

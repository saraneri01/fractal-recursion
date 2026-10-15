"""Training entry point for HRM-F C2, seed 0: reuses the C1 seed-0 configuration with the C2 architecture and stops at 4,886 optimiser updates.
"""
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
import torch
import yaml
from train.hrm import PretrainConfig, create_dataloader, init_train_state, train_batch
C1_CONFIG = Path(__file__).resolve().parents[1] / 'configs/hrm/fractal_c1_seed0.yaml'
C2_ARCH = 'hrm_fractal_c2@HierarchicalReasoningModel_ACTV1'
C2_SOURCE = Path(__file__).resolve().parents[1] / 'models/hrm_fractal_c2.py'
TARGET_STEP = 4886
OUT = Path(__file__).resolve().parents[1] / 'checkpoints/hrm/fractal_c2_seed0'
if OUT.exists():
    raise RuntimeError(f'{OUT} already exists. Refusing to overwrite.')
OUT.mkdir(parents=True)
with C1_CONFIG.open() as f:
    raw = yaml.safe_load(f)
source_arch = raw['arch']['name']
raw['arch']['name'] = C2_ARCH
raw['checkpoint_path'] = str(OUT)
config = PretrainConfig(**raw)
with (OUT / 'all_config.yaml').open('w') as f:
    yaml.safe_dump(config.model_dump(), f, sort_keys=False)
shutil.copy2(C2_SOURCE, OUT / C2_SOURCE.name)
source_sha256 = hashlib.sha256(C2_SOURCE.read_bytes()).hexdigest()
manifest = {'run': 'HRMF_C2_fullmacrograd_seed0_v1', 'source_config': str(C1_CONFIG), 'source_arch': source_arch, 'c2_arch': C2_ARCH, 'c2_source': str(C2_SOURCE), 'c2_source_sha256': source_sha256, 'seed': config.seed, 'target_actual_step': TARGET_STEP, 'purpose': 'Full training only. No evaluation. C2 differs from C1 only in gradient horizon through the final fractal macro-step.'}
with (OUT / 'manifest.json').open('w') as f:
    json.dump(manifest, f, indent=2)
torch.random.manual_seed(config.seed)
print('=' * 88)
print('HRM-F C2 FULL TRAINING — SEED 0')
print('=' * 88)
print('source C1 arch :', source_arch)
print('C2 arch        :', config.arch.name)
print('data           :', config.data_path)
print('seed           :', config.seed)
print('batch          :', config.global_batch_size)
print('base LR        :', config.lr)
print('warmup         :', config.lr_warmup_steps)
print('epochs config  :', config.epochs)
print('target step    :', TARGET_STEP)
print('output         :', OUT)
print('source SHA256  :', source_sha256)
print()
train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
train_loader, train_metadata = create_dataloader(config, 'train', rank=0, world_size=1, test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size)
train_state = init_train_state(config, train_metadata, world_size=1)
print('scheduler total_steps :', train_state.total_steps)
if train_state.total_steps != 5057:
    raise RuntimeError(f'Unexpected scheduler total_steps={train_state.total_steps}; expected 5057.')
num_params = sum((p.numel() for p in train_state.model.parameters()))
print('parameters            :', f'{num_params:,}')
if num_params != 27276290:
    raise RuntimeError(f'Unexpected parameter count: {num_params}')
train_state.model.train()
torch.cuda.reset_peak_memory_stats()
history_path = OUT / 'train_history.jsonl'
t0 = time.time()
last_step = 0
with history_path.open('w', buffering=1) as history_file:
    for set_name, batch, global_batch_size in train_loader:
        metrics = train_batch(config, train_state, batch, global_batch_size, rank=0, world_size=1)
        if metrics is None:
            continue
        step = int(train_state.step)
        last_step = step
        row = {'step': step}
        for k, v in metrics.items():
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(x):
                raise RuntimeError(f'NON-FINITE METRIC at step {step}: {k}={x}')
            row[k] = x
        history_file.write(json.dumps(row) + '\n')
        if step in (1, 20, 100) or step % 100 == 0 or step == TARGET_STEP:
            lm = row.get('train/lm_loss', float('nan'))
            acc = row.get('train/accuracy', float('nan'))
            count = row.get('train/count', float('nan'))
            act = row.get('train/steps', float('nan'))
            print(f'step {step:4d}/{TARGET_STEP} | lm_loss={lm:.6f} | acc={acc:.6f} | count={count:.3f} | ACT_steps={act:.3f}', flush=True)
        if step >= TARGET_STEP:
            break
if last_step != TARGET_STEP:
    raise RuntimeError(f'Training ended at step {last_step}; expected {TARGET_STEP}.')
torch.cuda.synchronize()
elapsed = time.time() - t0
peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
checkpoint = OUT / f'step_{TARGET_STEP}'
torch.save(train_state.model.state_dict(), checkpoint)
summary = {'architecture': C2_ARCH, 'seed': config.seed, 'parameter_count': num_params, 'scheduler_total_steps': train_state.total_steps, 'actual_final_step': last_step, 'elapsed_seconds': elapsed, 'seconds_per_step_including_compile': elapsed / TARGET_STEP, 'peak_cuda_gib': peak_gib, 'checkpoint': str(checkpoint), 'history': str(history_path), 'source_sha256': source_sha256}
with (OUT / 'training_summary.json').open('w') as f:
    json.dump(summary, f, indent=2)
print()
print('=' * 88)
print('FULL TRAINING COMPLETE')
print('=' * 88)
print(json.dumps(summary, indent=2))
print()
print('NO EVALUATION WAS RUN.')

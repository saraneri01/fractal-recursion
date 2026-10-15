"""Training of the final rapid HRM series (paper, Appendix A). Model A is the baseline, model C the nested variant.
"""
from __future__ import annotations
import argparse
import os
import random
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from adam_atan2_pytorch import AdamAtan2
from data.rapid.arc_dataset import load_arc_tasks, task_level_kfold, canvas_to_grid, PAD_ID
from data.rapid.task_dataset import ARCTaskConditionedDataset, build_task_query_examples, build_eval_sample
from models.rapid.hrm_baseline import FlatTaskConfig, FlatTaskRapid, count_params
from models.rapid.hrm_fractal import FractalTaskConfig, FractalTaskRapid
DATA_DIR = 'datasets/arc-agi-1/data/training'
OUTPUT_DIR = 'runs/rapid_hrm_final'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SPLIT_SEED = 0
BATCH_SIZE = 8
WEIGHT_DECAY = 0.1
CANVAS = 30
BEST_LR = {'A': 2.5e-05, 'C': 2.5e-05}
DEFAULT_N_SUP = 4
KEEP_LAST_N = 2

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_state(model_type, model, batch_size):
    if model_type == 'A':
        z_H, z_L = model.init_state(batch_size, DEVICE)
        return (z_H, z_L)
    else:
        z_H, z_M, z_L = model.init_state(batch_size, DEVICE)
        return (z_H, z_M, z_L)

def detach_state(z_state):
    return tuple((z.detach() for z in z_state))

def forward_one_segment(model_type, model, query_input, demo_inputs, demo_outputs, demo_mask, z_state):
    if model_type == 'A':
        z_H, z_L = z_state
        z_H, z_L, y_hat = model(query_input, demo_inputs, demo_outputs, demo_mask, z_H, z_L)
        return ((z_H, z_L), y_hat)
    else:
        z_H, z_M, z_L = z_state
        z_H, z_M, z_L, y_hat = model(query_input, demo_inputs, demo_outputs, demo_mask, z_H, z_M, z_L)
        return ((z_H, z_M, z_L), y_hat)

@torch.no_grad()
def evaluate(model_type, model, val_dict, n_sup, limit_tasks=None):
    import random
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_cpu_rng_state = torch.get_rng_state()
    torch_cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    model.eval()
    exact_count = 0
    correct_dimensions = 0
    real_correct = 0
    real_total = 0
    seed_everything(12345)
    items = list(val_dict.items())
    if limit_tasks:
        items = items[:limit_tasks]
    for task_id, task in items:
        test_pairs = task.get('test', [])
        if not test_pairs:
            raise RuntimeError(f'{task_id}: no test cases')
        case_exacts = []
        case_dims = []
        for test_index, pair in enumerate(test_pairs):
            if 'output' not in pair:
                raise RuntimeError(f'{task_id} test[{test_index}]: missing output')
            single_test_task = {'train': task['train'], 'test': [pair]}
            sample = build_eval_sample(single_test_task)
            query_input = sample['query_input'].to(DEVICE)
            demo_inputs = sample['demo_inputs'].to(DEVICE)
            demo_outputs = sample['demo_outputs'].to(DEVICE)
            demo_mask = sample['demo_mask'].to(DEVICE)
            target_canvas = sample['query_target_canvas']
            target_grid = sample['query_target_grid']
            z_state = init_state(model_type, model, 1)
            for _ in range(n_sup):
                z_state, y_hat = forward_one_segment(model_type, model, query_input, demo_inputs, demo_outputs, demo_mask, z_state)
            prediction = y_hat.argmax(dim=-1).reshape(CANVAS, CANVAS).cpu().numpy()
            prediction_grid = canvas_to_grid(prediction)
            same_shape = prediction_grid.shape == target_grid.shape
            exact = bool(same_shape and np.array_equal(prediction_grid, target_grid))
            case_dims.append(bool(same_shape))
            case_exacts.append(exact)
            real_mask = target_canvas != PAD_ID
            real_correct += int((prediction[real_mask] == target_canvas[real_mask]).sum())
            real_total += int(real_mask.sum())
        task_exact = all(case_exacts)
        task_correct_dims = all(case_dims)
        exact_count += int(task_exact)
        correct_dimensions += int(task_correct_dims)
    real_accuracy = 100.0 * real_correct / max(real_total, 1)
    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    torch.set_rng_state(torch_cpu_rng_state)
    if torch_cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(torch_cuda_rng_state)
    if was_training:
        model.train()
    else:
        model.eval()
    return (exact_count, len(items), real_accuracy, correct_dimensions)

def build_model(model_type, init_seed):
    seed_everything(init_seed)
    if model_type == 'A':
        cfg = FlatTaskConfig()
        model = FlatTaskRapid(cfg).to(DEVICE)
    else:
        cfg = FractalTaskConfig()
        model = FractalTaskRapid(cfg).to(DEVICE)
    return (model, cfg)

def ckpt_dir_for(model_type, init_seed):
    return os.path.join(OUTPUT_DIR, f'seed{init_seed}', f'{model_type}_ckpt')

def save_checkpoint(model, optimizer, step, curve, model_type, init_seed):
    out_dir = ckpt_dir_for(model_type, init_seed)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'step_{step}.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'curve': curve}, path)
    all_ckpts = sorted((f for f in os.listdir(out_dir) if f.startswith('step_') and f.endswith('.pt')), key=lambda f: int(f.split('_')[1].split('.')[0]))
    for old in all_ckpts[:-KEEP_LAST_N]:
        os.remove(os.path.join(out_dir, old))

def try_resume(model, optimizer, model_type, init_seed):
    out_dir = ckpt_dir_for(model_type, init_seed)
    if not os.path.isdir(out_dir):
        return (0, [])
    ckpts = sorted((f for f in os.listdir(out_dir) if f.startswith('step_') and f.endswith('.pt')), key=lambda f: int(f.split('_')[1].split('.')[0]))
    if not ckpts:
        return (0, [])
    path = os.path.join(out_dir, ckpts[-1])
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    print(f'  [resume] resumed from {path} at step {state['step']}')
    return (state['step'], state.get('curve', []))

def run_long(model_type, lr, n_sup, init_seed, train_dict, val_dict, max_steps, eval_every, eval_limit_tasks, resume):
    print(f'\n{'=' * 60}')
    print(f'LONG RUN WITH DEEP SUPERVISION -- MODEL {model_type}  (lr={lr:g}, N_SUP={n_sup}, init_seed={init_seed})')
    print(f'{'=' * 60}')
    model, cfg = build_model(model_type, init_seed)
    n_params = count_params(model)
    print(f'parameters: {n_params:,}')
    examples = build_task_query_examples(train_dict, include_official_test=True)
    print(f'training examples: {len(examples)}')
    seed_everything(init_seed)
    dataset = ARCTaskConditionedDataset(examples, augment=True, seed=init_seed)
    generator = torch.Generator()
    generator.manual_seed(init_seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0, drop_last=False)
    optimizer = AdamAtan2(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    data_iter = iter(loader)
    curve = []
    start_step = 0
    if resume:
        start_step, curve = try_resume(model, optimizer, model_type, init_seed)
        if start_step >= max_steps:
            print(f'  [resume] start_step ({start_step}) >= max_steps ({max_steps}): nothing to do.')
            return curve
    start_time = time.time()
    model.train()
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        query_input = batch['query_input'].to(DEVICE)
        target = batch['query_target'].to(DEVICE)
        demo_inputs = batch['demo_inputs'].to(DEVICE)
        demo_outputs = batch['demo_outputs'].to(DEVICE)
        demo_mask = batch['demo_mask'].to(DEVICE)
        z_state = init_state(model_type, model, query_input.shape[0])
        seg_losses = []
        for seg in range(n_sup):
            optimizer.zero_grad()
            z_state, y_hat = forward_one_segment(model_type, model, query_input, demo_inputs, demo_outputs, demo_mask, z_state)
            loss = F.cross_entropy(y_hat.reshape(-1, cfg.vocab_size), target.reshape(-1))
            loss.backward()
            optimizer.step()
            seg_losses.append(loss.item())
            z_state = detach_state(z_state)
        avg_seg_loss = sum(seg_losses) / len(seg_losses)
        if step % eval_every == 0 or step == max_steps:
            exact, total, real_acc, correct_dims = evaluate(model_type, model, val_dict, n_sup, eval_limit_tasks)
            elapsed = (time.time() - start_time) / 60
            curve.append((step, exact, total, real_acc, correct_dims))
            print(f'[{model_type} seed={init_seed}] step {step:5d}/{max_steps}  avg_seg_loss={avg_seg_loss:.4f}  held-out exact={exact}/{total}  real_acc={real_acc:.2f}%  correct_dims={correct_dims}/{total}  elapsed={elapsed:.1f} min', flush=True)
            save_checkpoint(model, optimizer, step, curve, model_type, init_seed)
    print(f'\nLearning curve -- MODEL {model_type} seed={init_seed} (lr={lr:g}, N_SUP={n_sup}):')
    print(f'{'step':>6} {'exact':>10} {'real_acc%':>10} {'correct_dims':>13}')
    for step, exact, total, real_acc, correct_dims in curve:
        print(f'{step:>6} {exact:>4}/{total:<5} {real_acc:>9.2f} {correct_dims:>7}/{total}')
    if len(curve) >= 3:
        accs = [c[3] for c in curve]
        deltas = [accs[i + 1] - accs[i] for i in range(len(accs) - 1)]
        print(f'\nChange in real_acc between evaluations: {[round(d, 2) for d in deltas]}')
        last3_avg_delta = sum((abs(d) for d in deltas[-3:])) / min(3, len(deltas))
        print(f'Mean absolute change over the last 3 evaluations: {last3_avg_delta:.2f} points')
        if last3_avg_delta < 1.0:
            print('-> PLATEAU signal: minimal change over recent evaluations.')
        else:
            print('-> STILL CHANGING: accuracy continues to change')
            print('   substantially -- additional training may be needed.')
    return curve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=250)
    parser.add_argument('--eval_limit_tasks', type=int, default=None)
    parser.add_argument('--model', type=str, choices=['A', 'C', 'both'], default='both')
    parser.add_argument('--n_sup', type=int, default=DEFAULT_N_SUP)
    parser.add_argument('--lr_override', type=float, default=None, help="Override BEST_LR for the model selected with --model (requires --model A or --model C, not 'both').")
    parser.add_argument('--init_seed', type=int, default=0, help='Seed for weight initialisation and data order. SPLIT_SEED stays fixed.')
    parser.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint for this model and seed.')
    args = parser.parse_args()
    if args.lr_override is not None and args.model == 'both':
        raise SystemExit("--lr_override requires --model A or --model C, not 'both'.")
    if args.resume and args.model == 'both':
        raise SystemExit("--resume requires --model A or --model C, not 'both'.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'device: {DEVICE}')
    print(f'max_steps={args.max_steps} (batch)  eval_every={args.eval_every}  N_SUP={args.n_sup}  init_seed={args.init_seed}  resume={args.resume}')
    print(f'-> total optimiser updates per model: {args.max_steps * args.n_sup}')
    lr_a = args.lr_override if args.lr_override is not None and args.model == 'A' else BEST_LR['A']
    lr_c = args.lr_override if args.lr_override is not None and args.model == 'C' else BEST_LR['C']
    print(f'Learning rates used: A={lr_a:g}  C={lr_c:g}')
    all_tasks = load_arc_tasks(DATA_DIR)
    all_task_ids = list(all_tasks.keys())
    train_ids, val_ids = next(task_level_kfold(all_task_ids, k=5, seed=SPLIT_SEED))
    train_dict = {t: all_tasks[t] for t in train_ids}
    val_dict = {t: all_tasks[t] for t in val_ids}
    print(f'train_tasks={len(train_dict)}  held_out_tasks={len(val_dict)}')
    curves = {}
    if args.model in ('A', 'both'):
        curves['A'] = run_long('A', lr_a, args.n_sup, args.init_seed, train_dict, val_dict, args.max_steps, args.eval_every, args.eval_limit_tasks, args.resume)
    if args.model in ('C', 'both'):
        curves['C'] = run_long('C', lr_c, args.n_sup, args.init_seed, train_dict, val_dict, args.max_steps, args.eval_every, args.eval_limit_tasks, args.resume)
    if 'A' in curves and 'C' in curves:
        print(f'\n{'=' * 60}')
        print(f'FINAL A vs C COMPARISON BY STEP (seed={args.init_seed})')
        print(f'{'=' * 60}')
        print(f'{'step':>6} {'A real_acc%':>12} {'C real_acc%':>12} {'gap (A-C)':>10}')
        for (step_a, _, _, acc_a, _), (step_c, _, _, acc_c, _) in zip(curves['A'], curves['C']):
            print(f'{step_a:>6} {acc_a:>12.2f} {acc_c:>12.2f} {acc_a - acc_c:>10.2f}')
if __name__ == '__main__':
    main()

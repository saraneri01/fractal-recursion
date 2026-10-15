"""Training and evaluation of the rapid TRM series (paper, Appendix A). Model A is the baseline, model C the nested variant.
"""
from __future__ import annotations
import argparse
import json
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
from data.rapid.arc_dataset import load_arc_tasks, canvas_to_grid, PAD_ID
from data.rapid.task_dataset import ARCTaskConditionedDataset, build_task_query_examples, build_eval_sample
from models.rapid.trm_baseline import TRMTaskConfig, TRMTaskBaseline, count_params
from models.rapid.trm_fractal import TRMFTaskConfig, TRMFTaskModel
TRAIN_DIR = 'datasets/arc-agi-1/data/training'
TEST_DIR = 'datasets/arc-agi-1/data/evaluation'
OUTPUT_DIR = 'runs/rapid_trm'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_STEPS = 20000
BATCH_SIZE = 8
WEIGHT_DECAY = 0.1
BEST_LR = {'A': 1e-05, 'C': 3e-05}
SEEDS = [0, 1, 2, 3, 4]
CANVAS = 30

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(model_type: str, init_seed: int):
    seed_everything(init_seed)
    if model_type == 'A':
        cfg = TRMTaskConfig()
        model = TRMTaskBaseline(cfg).to(DEVICE)
        core_calls = cfg.H_cycles * cfg.L_cycles
    else:
        cfg = TRMFTaskConfig()
        model = TRMFTaskModel(cfg).to(DEVICE)
        core_calls = cfg.H_cycles * cfg.M_cycles * cfg.L_cycles
    return (model, cfg, core_calls)

@torch.no_grad()
def evaluate_on_test(model_type, model, test_tasks):
    model.eval()
    per_task_results = {}
    real_correct_total, real_total_total = (0, 0)
    for task_id, task in test_tasks.items():
        if 'output' not in task['test'][0]:
            continue
        sample = build_eval_sample(task)
        query_input = sample['query_input'].to(DEVICE)
        demo_inputs = sample['demo_inputs'].to(DEVICE)
        demo_outputs = sample['demo_outputs'].to(DEVICE)
        demo_mask = sample['demo_mask'].to(DEVICE)
        target_canvas = sample['query_target_canvas']
        target_grid = sample['query_target_grid']
        if model_type == 'A':
            z_H, z_L = model.init_state(1, DEVICE)
            z_H, z_L, y_hat = model(query_input, demo_inputs, demo_outputs, demo_mask, z_H, z_L)
        else:
            z_H, z_M, z_L = model.init_state(1, DEVICE)
            z_H, z_M, z_L, y_hat = model(query_input, demo_inputs, demo_outputs, demo_mask, z_H, z_M, z_L)
        prediction = y_hat.argmax(dim=-1).reshape(CANVAS, CANVAS).cpu().numpy()
        prediction_grid = canvas_to_grid(prediction)
        same_shape = prediction_grid.shape == target_grid.shape
        exact = bool(same_shape and np.array_equal(prediction_grid, target_grid))
        real_mask = target_canvas != PAD_ID
        real_correct = int((prediction[real_mask] == target_canvas[real_mask]).sum())
        real_total = int(real_mask.sum())
        real_correct_total += real_correct
        real_total_total += real_total
        per_task_results[task_id] = {'exact_match': exact, 'correct_dims': bool(same_shape), 'real_correct': real_correct, 'real_total': real_total}
    model.train()
    overall_real_acc = 100.0 * real_correct_total / max(real_total_total, 1)
    n_exact = sum((1 for r in per_task_results.values() if r['exact_match']))
    return (per_task_results, n_exact, len(per_task_results), overall_real_acc)

def train_and_evaluate(model_type, init_seed, train_dict, test_tasks):
    lr = BEST_LR[model_type]
    print(f'\n{'=' * 60}')
    print(f'MODEL {model_type}_trm  seed={init_seed}  lr={lr:g}  max_steps={MAX_STEPS}')
    print(f'{'=' * 60}')
    model, cfg, core_calls = build_model(model_type, init_seed)
    n_params = count_params(model)
    print(f'parameters: {n_params:,}   core calls (legacy estimate): {core_calls}')
    examples = build_task_query_examples(train_dict, include_official_test=True)
    print(f'training examples: {len(examples)}')
    seed_everything(init_seed)
    dataset = ARCTaskConditionedDataset(examples, augment=True, seed=init_seed)
    generator = torch.Generator()
    generator.manual_seed(init_seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0, drop_last=False)
    optimizer = AdamAtan2(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    data_iter = iter(loader)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    model.train()
    for step in range(1, MAX_STEPS + 1):
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
        optimizer.zero_grad()
        if model_type == 'A':
            z_H, z_L = model.init_state(query_input.shape[0], DEVICE)
            z_H, z_L, y_hat = model(query_input, demo_inputs, demo_outputs, demo_mask, z_H, z_L)
        else:
            z_H, z_M, z_L = model.init_state(query_input.shape[0], DEVICE)
            z_H, z_M, z_L, y_hat = model(query_input, demo_inputs, demo_outputs, demo_mask, z_H, z_M, z_L)
        loss = F.cross_entropy(y_hat.reshape(-1, cfg.vocab_size), target.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 2000 == 0 or step == MAX_STEPS:
            elapsed = (time.time() - start_time) / 60
            print(f'  step {step:5d}/{MAX_STEPS}  loss={loss.item():.4f}  elapsed={elapsed:.1f} min', flush=True)
    train_elapsed_min = (time.time() - start_time) / 60
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1000000.0 if torch.cuda.is_available() else None
    print(f'\nTraining completed in {train_elapsed_min:.1f} min. Evaluating on the evaluation set...')
    per_task_results, n_exact, n_total, real_acc = evaluate_on_test(model_type, model, test_tasks)
    print(f'RESULT -- {model_type}_trm seed={init_seed}: exact={n_exact}/{n_total}  real_acc={real_acc:.2f}%  time={train_elapsed_min:.1f}min  peak_mem={peak_mem_mb}MB')
    return {'model': model_type, 'seed': init_seed, 'lr': lr, 'max_steps': MAX_STEPS, 'n_params': n_params, 'core_calls_per_forward': core_calls, 'train_elapsed_min': train_elapsed_min, 'peak_gpu_memory_mb': peak_mem_mb, 'n_exact': n_exact, 'n_total': n_total, 'real_acc': real_acc, 'per_task_results': per_task_results}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'device: {DEVICE}')
    print(f'seeds to run: {args.seeds}')
    print(f'LR: {BEST_LR}   max_steps: {MAX_STEPS}   batch: {BATCH_SIZE}')
    train_dict = load_arc_tasks(TRAIN_DIR)
    test_tasks = load_arc_tasks(TEST_DIR)
    print(f'train_tasks (all): {len(train_dict)}   test_tasks (evaluation): {len(test_tasks)}')
    all_results = []
    for seed in args.seeds:
        for model_type in ['A', 'C']:
            r = train_and_evaluate(model_type, seed, train_dict, test_tasks)
            all_results.append(r)
            out_path = os.path.join(OUTPUT_DIR, 'all_results_trm.json')
            with open(out_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f'[saved: {out_path}]')
    print(f'\n{'=' * 60}')
    print('FINAL SUMMARY (TRM family)')
    print(f'{'=' * 60}')
    print(f'{'model':>6} {'seed':>6} {'exact':>10} {'real_acc%':>10} {'time(min)':>11} {'peak_mem(MB)':>13}')
    for r in all_results:
        mem_str = f'{r['peak_gpu_memory_mb']:.0f}' if r['peak_gpu_memory_mb'] else 'n/a'
        print(f'{r['model']:>6} {r['seed']:>6} {r['n_exact']:>4}/{r['n_total']:<5} {r['real_acc']:>9.2f} {r['train_elapsed_min']:>11.1f} {mem_str:>13}')
    print(f'\nFull results saved to: {os.path.join(OUTPUT_DIR, 'all_results_trm.json')}')
if __name__ == '__main__':
    main()

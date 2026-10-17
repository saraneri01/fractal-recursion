"""Evaluate an official-codebase URM checkpoint on ARC-AGI-1 (paper, Tables 3 and 5).

Each run name selects the training configuration and the evaluated checkpoint.
Evaluation uses the held-out test queries, a horizon of 16 loops and an
evaluation batch size of 48. Metrics are printed and written to
runs/urm_evaluations/<run>/metrics.json. With --save-predictions, the inputs,
labels, predictions, puzzle identifiers and halting logits of every evaluated
sequence are also saved as .npy arrays, so that ARC/pass@k can be recomputed
without further inference (see scripts/recompute_urm_passk.py).

Usage:
    python -m eval.urm --run baseline_seed0_60ep --save-predictions
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import train.urm as pt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/urm/arc-aug-1000-eval-v1'
EVAL_BATCH_SIZE = 48
EVAL_LOOPS = 16

# run name -> evaluated checkpoint step
RUNS = {
    'baseline_seed0_60ep': 4807,
    'fractal_seed0_60ep': 4807,
    'baseline_seed0_120ep': 9614,
    'fractal_seed0_120ep': 9614,
    'fractal_damped_a025_seed0_30ep': 2405,
    'fractal_damped_a025_seed0_60ep': 4807,
}


def to_jsonable(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


# Runs whose stored checkpoint step was verified when loading.
CHECKED_STEPS = {'baseline_seed0_60ep', 'fractal_seed0_60ep', 'baseline_seed0_120ep', 'fractal_seed0_120ep'}


def attach_prediction_recorder(evaluator, data_path, out):
    """Save every evaluated sequence as compact .npy arrays while the evaluator runs."""
    shape = np.load(Path(data_path) / 'test/all__inputs.npy', mmap_mode='r').shape
    n_sequences = shape[0]
    specs = (
        ('inputs', np.uint8, shape),
        ('labels', np.int8, shape),
        ('preds', np.uint8, shape),
        ('puzzle_identifiers', np.int32, (n_sequences,)),
        ('q_halt_logits', np.float32, (n_sequences,)),
    )
    arrays = {name: np.lib.format.open_memmap(out / f'{name}.npy', mode='w+', dtype=dtype, shape=dims)
              for name, dtype, dims in specs}
    state = {'count': 0}
    original_update = evaluator.update_batch

    def update(batch, preds):
        values = {
            'inputs': batch['inputs'].detach().cpu().numpy(),
            'labels': batch['labels'].detach().cpu().numpy(),
            'puzzle_identifiers': batch['puzzle_identifiers'].detach().cpu().numpy(),
            'preds': preds['preds'].detach().cpu().numpy(),
            'q_halt_logits': preds['q_halt_logits'].detach().cpu().numpy(),
        }
        valid = (values['labels'] != -100).any(axis=1)
        start = state['count']
        end = start + int(valid.sum())
        if end > n_sequences:
            raise RuntimeError('More evaluated sequences than expected')
        for name, array in arrays.items():
            array[start:end] = values[name][valid]
        state['count'] = end
        return original_update(batch, preds)

    evaluator.update_batch = update

    def finish():
        for array in arrays.values():
            array.flush()
        if state['count'] != n_sequences:
            raise RuntimeError(f"Incomplete predictions: {state['count']}/{n_sequences}")
        print(f"Saved predictions for {state['count']} sequences in {out}")

    return finish


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', required=True, choices=sorted(RUNS), help='Configuration and checkpoint to evaluate.')
    parser.add_argument('--save-predictions', action='store_true', help='Also save compact .npy arrays of all evaluated sequences.')
    args = parser.parse_args()

    step = RUNS[args.run]
    config_path = ROOT / f'configs/urm/{args.run}.yaml'
    checkpoint = ROOT / f'checkpoints/urm/{args.run}/step_{step}.pt'
    out = ROOT / f'runs/urm_evaluations/{args.run}'
    if out.exists():
        raise SystemExit(f'Refusing to overwrite existing directory: {out}')
    out.mkdir(parents=True)

    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    config = pt.PretrainConfig(**raw)
    config.data_path = str(DATA)
    config.load_checkpoint = str(checkpoint)
    config.load_optimizer_state = False
    config.checkpoint_path = str(out)
    config.ema = False  # saved checkpoints already contain the EMA weights when EMA was enabled
    config.loop_deltas = [0]
    config.global_batch_size = EVAL_BATCH_SIZE
    config.evaluators[0].name = 'urm_arc@ARC'
    torch.manual_seed(config.seed)

    print(f'===== URM EVALUATION: {args.run} =====')
    print('architecture:', config.arch.name)
    print('checkpoint:', checkpoint)
    print('data:', DATA)
    print('loops:', config.arch.loops)
    print('H_cycles:', config.arch.H_cycles)
    print('L_cycles:', config.arch.L_cycles)
    print('M_cycles:', getattr(config.arch, 'M_cycles', None))
    print('L_cycles_inner:', getattr(config.arch, 'L_cycles_inner', None))
    print('M_damping_alpha:', getattr(config.arch, 'M_damping_alpha', 1.0))
    print('num_layers:', config.arch.num_layers)
    print('batch size:', config.global_batch_size)

    _, train_metadata = pt.create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)
    eval_loader, eval_metadata = pt.create_dataloader(config, 'test', test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)
    evaluators = pt.create_evaluators(config, eval_metadata)
    train_state = pt.init_train_state(config, train_metadata, rank=0, world_size=1)
    train_state.model.eval()
    loop_config = pt._get_loop_config(train_state.model)
    if loop_config is not None:
        loop_config.loops = EVAL_LOOPS
    print('checkpoint loaded: step', train_state.step)
    if args.run in CHECKED_STEPS:
        assert train_state.step == step, f'Unexpected checkpoint step: {train_state.step}'
    finish_recording = None
    if args.save_predictions:
        config.eval_save_outputs = []
        finish_recording = attach_prediction_recorder(evaluators[0], config.data_path, out)

    print('===== START EVALUATION =====')
    metrics = pt.evaluate(config, train_state, eval_loader, eval_metadata, evaluators, rank=0, world_size=1, cpu_group=None)
    if finish_recording is not None:
        finish_recording()
    print('\n===== FINAL METRICS =====')
    for key in sorted(metrics):
        print(f'{key}: {metrics[key]}')

    with open(out / 'metrics.json', 'w') as f:
        json.dump(to_jsonable(metrics), f, indent=2, sort_keys=True)
    print('\nSaved:', out / 'metrics.json')


if __name__ == '__main__':
    main()

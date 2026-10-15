from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
CANVAS = 30
PAD_ID = 10
NUM_COLORS = 10
VOCAB_SIZE = NUM_COLORS + 1

def load_arc_tasks(data_dir: str) -> Dict[str, dict]:
    data_dir = Path(data_dir)
    tasks = {}
    for f in sorted(data_dir.glob('*.json')):
        with open(f) as fh:
            tasks[f.stem] = json.load(fh)
    return tasks

def grid_to_canvas(grid: List[List[int]], canvas: int=CANVAS, row_off: int=0, col_off: int=0) -> np.ndarray:
    arr = np.array(grid, dtype=np.int64)
    h, w = arr.shape
    if h > canvas or w > canvas:
        raise ValueError(f'grid {h}x{w} exceeds canvas {canvas}')
    out = np.full((canvas, canvas), PAD_ID, dtype=np.int64)
    out[row_off:row_off + h, col_off:col_off + w] = arr
    return out

def canvas_to_grid(canvas_arr: np.ndarray) -> np.ndarray:
    mask = canvas_arr != PAD_ID
    if not mask.any():
        return np.zeros((1, 1), dtype=np.int64)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return canvas_arr[r0:r1 + 1, c0:c1 + 1]

@dataclass
class Transform:
    dihedral: int
    color_perm: Tuple[int, ...]
    row_off: int = 0
    col_off: int = 0

def _apply_dihedral(arr: np.ndarray, d: int) -> np.ndarray:
    flip = d >= 4
    k = d % 4
    out = np.fliplr(arr) if flip else arr
    out = np.rot90(out, k=k)
    return out

def _invert_dihedral(arr: np.ndarray, d: int) -> np.ndarray:
    flip = d >= 4
    k = d % 4
    out = np.rot90(arr, k=-k % 4)
    out = np.fliplr(out) if flip else out
    return out

def sample_transform(rng: random.Random, allow_color_perm: bool=True, fix_color_zero: bool=True) -> Transform:
    d = rng.randrange(8)
    if allow_color_perm:
        colors = list(range(1, NUM_COLORS)) if fix_color_zero else list(range(NUM_COLORS))
        rng.shuffle(colors)
        perm = [0] + colors if fix_color_zero else colors
    else:
        perm = list(range(NUM_COLORS))
    return Transform(dihedral=d, color_perm=tuple(perm))

def apply_transform_to_grid(grid: List[List[int]], t: Transform) -> np.ndarray:
    arr = np.array(grid, dtype=np.int64)
    arr = _apply_dihedral(arr, t.dihedral)
    perm = np.array(t.color_perm, dtype=np.int64)
    arr = perm[arr]
    return arr

def place_in_canvas(arr: np.ndarray, rng: random.Random, canvas: int=CANVAS) -> Tuple[np.ndarray, int, int]:
    h, w = arr.shape
    row_off = rng.randrange(0, canvas - h + 1)
    col_off = rng.randrange(0, canvas - w + 1)
    out = np.full((canvas, canvas), PAD_ID, dtype=np.int64)
    out[row_off:row_off + h, col_off:col_off + w] = arr
    return (out, row_off, col_off)

def transform_to_canvas(grid: List[List[int]], t: Transform, rng: random.Random, canvas: int=CANVAS) -> np.ndarray:
    arr = apply_transform_to_grid(grid, t)
    out, t.row_off, t.col_off = place_in_canvas(arr, rng, canvas)
    return out

def invert_prediction(canvas_arr: np.ndarray, t: Transform) -> np.ndarray:
    cropped = canvas_to_grid(canvas_arr)
    inv_perm = np.zeros(NUM_COLORS, dtype=np.int64)
    for i, c in enumerate(t.color_perm):
        inv_perm[c] = i
    cropped_safe = np.clip(cropped, 0, NUM_COLORS - 1)
    un_colored = inv_perm[cropped_safe]
    un_dihedral = _invert_dihedral(un_colored, t.dihedral)
    return un_dihedral

class ARCExample:
    __slots__ = ('task_id', 'input_grid', 'output_grid')

    def __init__(self, task_id: str, input_grid, output_grid):
        self.task_id = task_id
        self.input_grid = input_grid
        self.output_grid = output_grid

def flatten_examples(tasks: Dict[str, dict], include_test: bool=True) -> List[ARCExample]:
    examples = []
    for task_id, task in tasks.items():
        for pair in task['train']:
            examples.append(ARCExample(task_id, pair['input'], pair['output']))
        if include_test:
            for pair in task['test']:
                if 'output' in pair:
                    examples.append(ARCExample(task_id, pair['input'], pair['output']))
    return examples

class ARCDataset(Dataset):

    def __init__(self, examples: List[ARCExample], task_to_id: Dict[str, int], augment: bool=True, seed: int=0, fix_color_zero: bool=True):
        self.examples = examples
        self.task_to_id = task_to_id
        self.augment = augment
        self.rng = random.Random(seed)
        self.fix_color_zero = fix_color_zero

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.augment:
            t = sample_transform(self.rng, fix_color_zero=self.fix_color_zero)
            x = transform_to_canvas(ex.input_grid, t, self.rng)
            t_out = Transform(dihedral=t.dihedral, color_perm=t.color_perm)
            y = transform_to_canvas(ex.output_grid, t_out, self.rng)
        else:
            x = grid_to_canvas(ex.input_grid)
            y = grid_to_canvas(ex.output_grid)
        return {'input': torch.from_numpy(x).long(), 'target': torch.from_numpy(y).long(), 'puzzle_id': torch.tensor(self.task_to_id[ex.task_id], dtype=torch.long), 'task_id': ex.task_id}

def build_task_to_id(task_ids: List[str]) -> Dict[str, int]:
    uniq = sorted(set(task_ids))
    return {t: i for i, t in enumerate(uniq)}

def task_level_kfold(task_ids: List[str], k: int=5, seed: int=0):
    uniq = sorted(set(task_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    folds = [uniq[i::k] for i in range(k)]
    for i in range(k):
        val = set(folds[i])
        train = [t for t in uniq if t not in val]
        yield (train, sorted(val))

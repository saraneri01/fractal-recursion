from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from data.rapid.arc_dataset import CANVAS, PAD_ID, grid_to_canvas, sample_transform, apply_transform_to_grid
MAX_DEMOS = 12

@dataclass
class TaskQueryExample:
    task_id: str
    demos: List[Tuple[List[List[int]], List[List[int]]]]
    query_input: List[List[int]]
    query_output: List[List[int]]

def _to_canvas_no_translation(grid, transform=None):
    if transform is None:
        return grid_to_canvas(grid)
    arr = apply_transform_to_grid(grid, transform)
    return grid_to_canvas(arr.tolist())

def build_task_query_examples(tasks: Dict[str, dict], include_official_test: bool=True) -> List[TaskQueryExample]:
    examples = []
    for task_id, task in tasks.items():
        labeled_pairs = [(pair['input'], pair['output']) for pair in task['train']]
        if include_official_test:
            for pair in task.get('test', []):
                if 'output' in pair:
                    labeled_pairs.append((pair['input'], pair['output']))
        if len(labeled_pairs) < 2:
            continue
        for query_index, (query_input, query_output) in enumerate(labeled_pairs):
            demos = [pair for i, pair in enumerate(labeled_pairs) if i != query_index]
            examples.append(TaskQueryExample(task_id=task_id, demos=demos, query_input=query_input, query_output=query_output))
    return examples

class ARCTaskConditionedDataset(Dataset):

    def __init__(self, examples: List[TaskQueryExample], augment: bool=True, seed: int=0, max_demos: int=MAX_DEMOS):
        self.examples = examples
        self.augment = augment
        self.max_demos = max_demos
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.augment:
            transform = sample_transform(self.rng, fix_color_zero=True)
        else:
            transform = None
        query_input = _to_canvas_no_translation(ex.query_input, transform)
        query_target = _to_canvas_no_translation(ex.query_output, transform)
        demo_inputs = np.full((self.max_demos, CANVAS, CANVAS), PAD_ID, dtype=np.int64)
        demo_outputs = np.full((self.max_demos, CANVAS, CANVAS), PAD_ID, dtype=np.int64)
        demo_mask = np.zeros((self.max_demos,), dtype=np.bool_)
        demos = ex.demos[:self.max_demos]
        for i, (demo_input, demo_output) in enumerate(demos):
            demo_inputs[i] = _to_canvas_no_translation(demo_input, transform)
            demo_outputs[i] = _to_canvas_no_translation(demo_output, transform)
            demo_mask[i] = True
        return {'query_input': torch.from_numpy(query_input).long(), 'query_target': torch.from_numpy(query_target).long(), 'demo_inputs': torch.from_numpy(demo_inputs).long(), 'demo_outputs': torch.from_numpy(demo_outputs).long(), 'demo_mask': torch.from_numpy(demo_mask), 'task_id': ex.task_id}

def build_eval_sample(task: dict, max_demos: int=MAX_DEMOS):
    test_pair = task['test'][0]
    demos = [(pair['input'], pair['output']) for pair in task['train']][:max_demos]
    demo_inputs = np.full((1, max_demos, CANVAS, CANVAS), PAD_ID, dtype=np.int64)
    demo_outputs = np.full((1, max_demos, CANVAS, CANVAS), PAD_ID, dtype=np.int64)
    demo_mask = np.zeros((1, max_demos), dtype=np.bool_)
    for i, (demo_input, demo_output) in enumerate(demos):
        demo_inputs[0, i] = grid_to_canvas(demo_input)
        demo_outputs[0, i] = grid_to_canvas(demo_output)
        demo_mask[0, i] = True
    query_input = grid_to_canvas(test_pair['input'])[None, ...]
    query_target_canvas = grid_to_canvas(test_pair['output'])
    query_target_grid = np.array(test_pair['output'], dtype=np.int64)
    return {'query_input': torch.from_numpy(query_input).long(), 'query_target_canvas': query_target_canvas, 'query_target_grid': query_target_grid, 'demo_inputs': torch.from_numpy(demo_inputs).long(), 'demo_outputs': torch.from_numpy(demo_outputs).long(), 'demo_mask': torch.from_numpy(demo_mask)}

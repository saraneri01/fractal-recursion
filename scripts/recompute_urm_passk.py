"""Recompute URM ARC/pass@k on the CPU from saved predictions (paper, Section 3).

No model or checkpoint is loaded. Each run folder must contain the arrays
written by `python -m eval.urm --run <run> --save-predictions` (inputs.npy,
preds.npy, puzzle_identifiers.npy, q_halt_logits.npy), the ARC submission in
evaluator_*/submission.json and either audit_summary.json or metrics.json.

Cropping, grid normalisation, query matching, candidate ranking and scoring are
implemented here separately from the ARC evaluator; the inverse geometric
transform reuses the dataset helper. The script checks coverage of all 400
tasks and 419 test queries, checks that the first two ranked candidates match
the submission, and compares ARC/pass@k with the evaluator for k = 1, 2, 100,
1000. ARC/pass@k is the mean over tasks of the fraction of test queries solved
within the first k candidates; the number of entirely solved tasks is also
reported.

Usage:
    python -m scripts.recompute_urm_passk \
        --runs runs/urm_evaluations/baseline_seed0_60ep runs/urm_evaluations/fractal_seed0_60ep
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import re
import json
import math
import hashlib
from pathlib import Path
import numpy as np

from data.urm.common import inverse_dihedral_transform

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Recompute URM ARC/pass@k from saved predictions.")
parser.add_argument("--runs", nargs="+", type=Path, required=True, help="Evaluation folders to check.")
parser.add_argument("--data", type=Path, default=REPO / "datasets/urm/arc-aug-1000-eval-v1")
parser.add_argument("--raw", type=Path, default=REPO / "datasets/arc-agi-1/data/evaluation")
parser.add_argument("--expected-step", type=int, default=4807)
parser.add_argument("--expected-sequences", type=int, default=385814)
args = parser.parse_args()

DATA = args.data
RAW = args.raw
KS = (1, 2, 100, 1000)
FOLDERS = args.runs

print("SCRIPT:", Path(__file__).resolve())
print("SCRIPT SHA256:", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print("Saved data only; no training or inference.", flush=True)

identifiers = json.loads((DATA / "identifiers.json").read_text())
dataset_tasks = json.loads((DATA / "test_puzzles.json").read_text())
raw = {p.stem: json.loads(p.read_text()) for p in RAW.glob("*.json")}
assert len(raw) == 400
assert set(dataset_tasks) == set(raw)
assert sum(len(p["test"]) for p in raw.values()) == 419
for task in raw:
    assert dataset_tasks[task]["test"] == raw[task]["test"], task


def key(grid):
    a = np.asarray(grid)
    assert a.ndim == 2
    assert np.all(np.isfinite(a))
    assert np.all(a == np.floor(a))
    assert np.all((a >= 0) & (a <= 9))
    a = np.ascontiguousarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def crop(tokens):
    a = np.asarray(tokens, dtype=np.int64).reshape(30, 30)
    invalid = (a < 2) | (a > 11)
    widths = np.where(invalid.any(axis=1), invalid.argmax(axis=1), 30)
    widths = np.minimum.accumulate(widths)
    areas = widths * np.arange(1, 31)
    row = int(areas.argmax())
    if areas[row] == 0:
        return np.empty((0, 0), dtype=np.int64)
    return a[:row + 1, :int(widths[row])] - 2


plans = {}


def decode(identifier, tokens):
    identifier = int(identifier)
    if identifier not in plans:
        name = identifiers[identifier]
        match = re.fullmatch(r"(.+)_t([0-7])_([0-9]{10})", name)
        if match:
            task, transform, permutation = match.groups()
        elif "|||" in name:
            task, transform, permutation = name.rsplit("|||", 2)
            transform = transform.removeprefix("t")
        else:
            assert name in raw, f"Unrecognised identifier: {name}"
            plans[identifier] = (name, None, None)

        if identifier not in plans:
            permutation = np.array([int(c) for c in permutation])
            assert sorted(permutation.tolist()) == list(range(10))
            transform = int(transform)
            assert 0 <= transform <= 7
            assert task in raw
            plans[identifier] = (task, transform, np.argsort(permutation))

    task, transform, inverse_colors = plans[identifier]
    grid = crop(tokens)
    if transform is not None:
        grid = inverse_colors[inverse_dihedral_transform(grid, transform)]
    return task, grid


lookup, targets = {}, {}
for task, puzzle in raw.items():
    for index, pair in enumerate(puzzle["test"]):
        lookup.setdefault((task, key(pair["input"])), []).append((task, index))
        targets[(task, index)] = key(pair["output"])

print("Raw data checked: 400 tasks and 419 queries.", flush=True)

for folder in FOLDERS:
    print("\n=== RUN:", folder.name, "===", flush=True)
    print("FOLDER:", folder)
    if (folder / "audit_summary.json").exists():
        summary = json.loads((folder / "audit_summary.json").read_text())
        metrics = summary["metrics"]
        assert summary["complete"] is True
        assert summary["checkpoint_step"] == args.expected_step
        assert summary["sequences"] == args.expected_sequences
    else:
        metrics = json.loads((folder / "metrics.json").read_text())

    arrays = {
        name: np.load(folder / f"{name}.npy",
                      mmap_mode="r", allow_pickle=False)
        for name in ("inputs", "preds", "puzzle_identifiers", "q_halt_logits")
    }
    n = len(arrays["preds"])
    assert n == args.expected_sequences
    assert all(len(a) == n for a in arrays.values())

    votes = {query: {} for query in targets}
    seen, unmatched = set(), 0

    for i in range(n):
        identifier = int(arrays["puzzle_identifiers"][i])
        assert identifier != 0, f"Blank identifier: row {i}"
        task, original_input = decode(identifier, arrays["inputs"][i])
        _, prediction = decode(identifier, arrays["preds"][i])
        queries = lookup.get((task, key(original_input)), [])

        if not queries:
            unmatched += 1
        else:
            x = float(arrays["q_halt_logits"][i])
            assert math.isfinite(x)
            if x >= 0:
                q = 1.0 / (1.0 + math.exp(-x))
            else:
                e = math.exp(x)
                q = e / (1.0 + e)
            log_q = -float(np.logaddexp(0.0, -x))
            candidate = key(prediction)
            for query in queries:
                seen.add(query)
                stats = votes[query].setdefault(candidate, [0, 0.0, -math.inf])
                stats[0] += 1
                stats[1] += q
                stats[2] = max(stats[2], log_q)

        if (i + 1) % 50000 == 0:
            print(f"Rows processed: {i+1}/{n}", flush=True)

    assert unmatched == 0, f"Unmatched sequences: {unmatched}"
    assert len(seen) == 419
    assert len({task for task, _ in seen}) == 400
    print("Coverage: 400 tasks, 419 queries; unmatched sequences: 0")

    ranks, top_two = {}, {}
    for query, candidates in votes.items():
        ordered = sorted(
            candidates,
            key=lambda c: (
                candidates[c][0],
                candidates[c][1] / candidates[c][0],
                candidates[c][2],
            ),
            reverse=True,
        )
        assert ordered, f"Query without candidates: {query}"
        target = targets[query]
        ranks[query] = ordered.index(target) + 1 if target in candidates else None
        top_two[query] = (ordered + [ordered[0]])[:2]

    paths = list(folder.glob("evaluator_*/submission.json"))
    assert len(paths) == 1
    submission = json.loads(paths[0].read_text())
    assert set(submission) == set(raw)
    assert sum(len(a) for a in submission.values()) == 419

    for task, answers in submission.items():
        assert len(answers) == len(raw[task]["test"])
        for index, answer in enumerate(answers):
            for attempt in (1, 2):
                assert key(answer[f"attempt_{attempt}"]) == (
                    top_two[(task, index)][attempt - 1]
                ), f"Submission mismatch: {task}, query {index}, attempt {attempt}"
    print("PASS: the first two candidates match the submission.")

    mismatches = []
    for k in KS:
        query_count, solved_tasks, macro_sum = 0, 0, 0.0
        for task, puzzle in raw.items():
            correct = 0
            for index in range(len(puzzle["test"])):
                rank = ranks[(task, index)]
                if rank is not None and rank <= k:
                    correct += 1
                    print(f"  @{k}: task={task}, query_index={index}, rank={rank}")
            query_count += correct
            solved_tasks += correct == len(puzzle["test"])
            macro_sum += correct / len(puzzle["test"])

        score = macro_sum / 400
        recorded = float(metrics[f"ARC/pass@{k}"])
        matches = abs(score - recorded) < 1e-12
        print(
            f"k={k}: mean_per_task={score:.12g} ({100*score:.8g}%); "
            f"queries={query_count}/419; "
            f"entirely_solved_tasks={solved_tasks}/400; "
            f"evaluator={recorded:.12g}; match={matches}",
            flush=True,
        )
        if not matches:
            mismatches.append(k)

    assert not mismatches, f"MISMATCHES AT K: {mismatches}"
    print("PASS: all four scores match.")
    del arrays, votes, ranks, top_two

print("\nCHECK COMPLETE: no model loaded, no training or inference.")

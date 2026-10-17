"""Recompute TRM ARC/pass@1 and ARC/pass@2 from the saved submissions (paper, Section 3).

No model or checkpoint is loaded. Each submission lists, for every ARC-AGI-1
evaluation task and test query, the first two ranked candidates
(`attempt_1`, `attempt_2`). A query is solved at k = 1 if `attempt_1` equals the
expected output, and at k = 2 if either attempt does. Grids are compared after
conversion to a common integer type. ARC/pass@k is the mean over tasks of the
fraction of solved test queries; the number of entirely solved tasks is also
reported. The recomputed scores are compared with the evaluator metrics stored
next to each submission (`<run>_metrics.json`).

Larger k cannot be recomputed from the submissions, which contain two
candidates per query.

Usage:
    python -m scripts.recompute_trm_pass12
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Recompute TRM ARC/pass@1 and ARC/pass@2 from saved submissions.")
parser.add_argument("--submissions", nargs="+", type=Path, default=[
    Path("results/trm_official/baseline_seed0_submission.json"),
    Path("results/trm_official/fractal_seed0_submission.json"),
])
parser.add_argument("--raw", type=Path, default=REPO / "datasets/arc-agi-1/data/evaluation")
args = parser.parse_args()


def key(grid):
    """Integer-normalised grid key; malformed or empty predictions never match a target."""
    a = np.asarray(grid)
    if a.ndim != 2 or a.size == 0 or not (np.all(a == np.floor(a)) and np.all((a >= 0) & (a <= 9))):
        return ("invalid", json.dumps(grid))
    a = np.ascontiguousarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


print("SCRIPT SHA256:", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
print("Saved submissions only; no model loaded, no training or inference.")

raw = {p.stem: json.loads(p.read_text()) for p in args.raw.glob("*.json")}
assert len(raw) == 400
assert sum(len(t["test"]) for t in raw.values()) == 419
print("Raw data checked: 400 tasks and 419 queries.")

for path in args.submissions:
    name = path.name.removesuffix("_submission.json")
    print(f"\n=== RUN: {name} ===")
    print("SUBMISSION:", path, "SHA256:", hashlib.sha256(path.read_bytes()).hexdigest())
    submission = json.loads(path.read_text())
    metrics = json.loads(path.with_name(f"{name}_metrics.json").read_text())

    assert set(submission) == set(raw), "Submission tasks differ from the evaluation tasks"
    assert sum(len(a) for a in submission.values()) == 419
    print("Coverage: 400 tasks, 419 queries.")

    mismatches = []
    for k in (1, 2):
        queries, solved_tasks, macro_sum = 0, 0, 0.0
        for task, puzzle in raw.items():
            answers = submission[task]
            assert len(answers) == len(puzzle["test"])
            correct = 0
            for index, (answer, pair) in enumerate(zip(answers, puzzle["test"])):
                target = key(pair["output"])
                assert target[0] != "invalid", f"Invalid target grid: {task}"
                attempts = [answer["attempt_1"]] + ([answer["attempt_2"]] if k == 2 else [])
                if any(key(a) == target for a in attempts):
                    correct += 1
                    print(f"  @{k}: task={task}, query_index={index}")
            queries += correct
            solved_tasks += correct == len(puzzle["test"])
            macro_sum += correct / len(puzzle["test"])
        score = macro_sum / 400
        recorded = float(metrics[f"ARC/pass@{k}"])
        match = abs(score - recorded) < 1e-12
        print(f"k={k}: mean_per_task={score:.12g} ({100*score:.8g}%); queries={queries}/419; "
              f"entirely_solved_tasks={solved_tasks}/400; evaluator={recorded:.12g}; match={match}")
        if not match:
            mismatches.append(k)
    assert not mismatches, f"MISMATCHES AT K: {mismatches}"
    print("PASS: both scores match.")

print("\nCHECK COMPLETE: no model loaded, no training or inference.")

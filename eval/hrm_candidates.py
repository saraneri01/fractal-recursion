"""Original-query and original-task scores at k for saved HRM predictions (paper, Table 2 and Section 4).
"""
import os
import json
import csv
import argparse
from glob import glob

import numpy as np
import torch

from data.hrm.common import inverse_dihedral_transform


PAD_PUZZLE_IDENTIFIER = 0


def grid_hash(grid):
    # Exact key, independent of the original integer dtype.
    grid = np.ascontiguousarray(grid, dtype=np.int64)
    return (grid.shape, grid.tobytes())


def inverse_aug(name, grid):
    """
    Undo the augmentation encoded in the puzzle identifier.
    This follows the official HRM arc_eval.ipynb.
    """
    if "_" not in name:
        return grid

    trans_id, perm = name.split("_")[-2:]
    trans_id = int(trans_id[1:])

    inv_perm = np.argsort([int(x) for x in perm])

    return inv_perm[inverse_dihedral_transform(grid, trans_id)]


def crop(grid):
    """
    Recover the actual ARC grid from the 30x30 padded representation.
    """
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)

    nr, nc = grid.shape
    num_c = nc

    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]

            if (x < 2) or (x > 11):
                num_c = c - 1
                break

        area = num_r * num_c

        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return grid[:max_size[0], :max_size[1]] - 2


def load_predictions(dataset_path, checkpoint_path):
    with open(os.path.join(dataset_path, "identifiers.json")) as f:
        identifier_map = json.load(f)

    files = sorted(glob(f"{checkpoint_path}_all_preds.*"))
    if not files:
        raise FileNotFoundError("No saved prediction files found")

    keys = ("inputs", "labels", "puzzle_identifiers", "q_halt_logits")
    chunks = {key: [] for key in keys}
    chunks["pred_tokens"] = []

    for filename in files:
        print("Reading:", filename, flush=True)
        data = torch.load(
            filename, map_location="cpu",
            weights_only=False, mmap=True,
        )
        n = len(data["puzzle_identifiers"])
        assert all(len(data[key]) == n for key in keys)
        assert len(data["logits"]) == n

        for begin in range(0, n, 2048):
            end = min(begin + 2048, n)
            mask = data["puzzle_identifiers"][begin:end] != PAD_PUZZLE_IDENTIFIER
            if not bool(mask.any()):
                continue

            for key in keys:
                chunks[key].append(data[key][begin:end][mask])

            tokens = data["logits"][begin:end].argmax(-1).to(torch.int32)
            chunks["pred_tokens"].append(tokens[mask])

            if begin % 65536 == 0:
                print(f"Decoded {end}/{n} rows", flush=True)

        del data

    result = {}
    for key in list(chunks):
        result[key] = torch.cat(chunks.pop(key), dim=0)

    return identifier_map, result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--out",
        required=True
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Loading HRM predictions")
    print("=" * 70)

    identifier_map, all_preds = load_predictions(
        args.dataset,
        args.checkpoint
    )

    print(
        f"Total evaluated puzzle instances: "
        f"{len(all_preds['puzzle_identifiers'])}"
    )

    # ------------------------------------------------------------
    # 1. Recover ORIGINAL test tasks and their ground-truth labels
    # ------------------------------------------------------------

    puzzle_labels = {}

    for identifier, inp, label in zip(
        all_preds["puzzle_identifiers"],
        all_preds["inputs"],
        all_preds["labels"]
    ):

        name = identifier_map[int(identifier)]

        # Only the original, non-augmented version is used
        # to define the ground truth.
        if "_" in name:
            continue

        inp = crop(inp.numpy())
        label = crop(label.numpy())

        input_hash = grid_hash(inp)
        label_hash = grid_hash(label)

        puzzle_labels.setdefault(name, {})

        if input_hash in puzzle_labels[name]:
            raise RuntimeError(
                f"Duplicate original test input for task {name}"
            )

        puzzle_labels[name][input_hash] = label_hash

    print(
        f"Original ARC tasks recovered: "
        f"{len(puzzle_labels)}"
    )

    total_test_pairs = sum(
        len(x) for x in puzzle_labels.values()
    )

    print(
        f"Original test pairs recovered: "
        f"{total_test_pairs}"
    )

    # ------------------------------------------------------------
    # 2. Collect ALL augmented predictions
    # ------------------------------------------------------------

    predictions = all_preds["pred_tokens"]
    q_values = all_preds["q_halt_logits"].sigmoid()

    pred_answers = {}

    for identifier, inp, pred, q in zip(
        all_preds["puzzle_identifiers"],
        all_preds["inputs"],
        predictions,
        q_values
    ):

        name = identifier_map[int(identifier)]

        # Original task name without augmentation suffix.
        orig_name = name.split("_")[0]

        # Return augmented input to original ARC space.
        input_original = inverse_aug(
            name,
            crop(inp.numpy())
        )

        input_hash = grid_hash(input_original)

        # This protects us against unexpected identifiers.
        if orig_name not in puzzle_labels:
            raise RuntimeError(f"Unknown original task: {name}")

        if input_hash not in puzzle_labels[orig_name]:
            raise RuntimeError(f"Unmatched original test input: {name}")

        # Return prediction to original ARC space.
        pred_original = inverse_aug(
            name,
            crop(pred.numpy())
        )

        pred_hash = grid_hash(pred_original)

        pred_answers.setdefault(
            orig_name,
            {}
        ).setdefault(
            input_hash,
            []
        ).append(
            (pred_hash, float(q.item()))
        )

    # ------------------------------------------------------------

    # Diagnostic: ranked candidates and availability of exact answers.
    ks = [1, 2, 10, 100, 1000]
    pair_hits = {k: 0 for k in ks}
    task_hits = {k: 0 for k in ks}
    oracle_pairs = 0
    oracle_tasks = 0

    for task_name, tests in puzzle_labels.items():
        task_correct = {k: 0 for k in ks}
        task_oracle = 0

        for input_key, target_key in tests.items():
            candidates = pred_answers[task_name][input_key]
            if not candidates:
                raise RuntimeError(f"No candidates for {task_name}")

            counts = {}
            for prediction_key, q in candidates:
                entry = counts.setdefault(prediction_key, [0, 0.0])
                entry[0] += 1
                entry[1] += q

            ranked = sorted(
                counts,
                key=lambda key: (
                    counts[key][0],
                    counts[key][1] / counts[key][0],
                ),
                reverse=True,
            )

            for k in ks:
                correct = target_key in ranked[:k]
                pair_hits[k] += int(correct)
                task_correct[k] += int(correct)

            available = target_key in counts
            oracle_pairs += int(available)
            task_oracle += int(available)

        for k in ks:
            task_hits[k] += int(task_correct[k] == len(tests))
        oracle_tasks += int(task_oracle == len(tests))

    diagnostic = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "n_tasks": len(puzzle_labels),
        "n_pairs": total_test_pairs,
        "correct_pairs_at_k": pair_hits,
        "solved_tasks_at_k": task_hits,
        "oracle_correct_pairs": oracle_pairs,
        "oracle_solved_tasks": oracle_tasks,
    }

    diagnostic_path = args.out + ".candidate_metrics.json"
    os.makedirs(os.path.dirname(diagnostic_path) or ".", exist_ok=True)
    with open(diagnostic_path, "x") as f:
        json.dump(diagnostic, f, indent=2)

    print("\\n===== CANDIDATE DIAGNOSTICS =====", flush=True)
    print(json.dumps(diagnostic, indent=2), flush=True)
    print("Saved:", diagnostic_path, flush=True)

    # 3. Official-style aggregation
    # ------------------------------------------------------------
    #
    # For each original test pair:
    #   - group identical de-augmented predictions
    #   - count their frequency
    #   - use mean q_halt as tie-breaker
    #   - select the highest-ranked prediction
    #
    # This corresponds to K=1 in the official arc_eval.ipynb.
    # ------------------------------------------------------------

    task_rows = []

    exact_test_pairs = 0
    solved_tasks = 0
    total_pairs = 0

    prediction_counts = []

    for task_name, tests in puzzle_labels.items():

        correct_pairs = 0

        for input_hash, label_hash in tests.items():

            candidates = (
                pred_answers
                .get(task_name, {})
                .get(input_hash, [])
            )

            if not candidates:
                raise RuntimeError(
                    f"No predictions found for "
                    f"task={task_name}, "
                    f"input_hash={input_hash}"
                )

            prediction_counts.append(len(candidates))

            # prediction_hash -> [frequency, q_sum]
            stats = {}

            for pred_hash, q in candidates:

                if pred_hash not in stats:
                    stats[pred_hash] = [0, 0.0]

                stats[pred_hash][0] += 1
                stats[pred_hash][1] += q

            ranked = []

            for pred_hash, (count, q_sum) in stats.items():

                mean_q = q_sum / count

                ranked.append(
                    (
                        pred_hash,
                        count,
                        mean_q
                    )
                )

            # Frequency first, q_halt second.
            ranked.sort(
                key=lambda x: (x[1], x[2]),
                reverse=True
            )

            best_prediction = ranked[0][0]

            correct = best_prediction == label_hash

            correct_pairs += int(correct)
            exact_test_pairs += int(correct)
            total_pairs += 1

        task_exact = (
            correct_pairs == len(tests)
        )

        solved_tasks += int(task_exact)

        task_rows.append({
            "task": task_name,
            "test_pairs": len(tests),
            "correct_pairs": correct_pairs,
            "task_exact_match": int(task_exact)
        })

    # ------------------------------------------------------------
    # 4. Save task-level audit
    # ------------------------------------------------------------

    task_rows.sort(
        key=lambda x: x["task"]
    )

    os.makedirs(
        os.path.dirname(args.out) or ".",
        exist_ok=True
    )

    with open(
        args.out,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "test_pairs",
                "correct_pairs",
                "task_exact_match"
            ]
        )

        writer.writeheader()
        writer.writerows(task_rows)

    # ------------------------------------------------------------
    # 5. Final results
    # ------------------------------------------------------------

    task_accuracy = (
        solved_tasks / len(puzzle_labels)
    )

    pair_accuracy = (
        exact_test_pairs / total_pairs
    )

    print()
    print("=" * 70)
    print("ARC-AGI-1 HRM OFFICIAL-STYLE PASS@1")
    print("=" * 70)

    print(
        f"Original ARC tasks:       "
        f"{len(puzzle_labels)}"
    )

    print(
        f"Original test pairs:      "
        f"{total_pairs}"
    )

    print(
        f"Exact test pairs:         "
        f"{exact_test_pairs}/{total_pairs}"
    )

    print(
        f"Pair exact accuracy:      "
        f"{pair_accuracy:.6f} "
        f"({100 * pair_accuracy:.3f}%)"
    )

    print(
        f"Solved tasks:             "
        f"{solved_tasks}/{len(puzzle_labels)}"
    )

    print(
        f"Task exact-match pass@1:  "
        f"{task_accuracy:.6f} "
        f"({100 * task_accuracy:.3f}%)"
    )

    print(
        f"Augmented predictions/"
        f"original test pair: "
        f"min={min(prediction_counts)}, "
        f"max={max(prediction_counts)}, "
        f"mean={np.mean(prediction_counts):.2f}"
    )

    print()
    print(
        f"Per-task audit saved to:\n"
        f"{args.out}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()


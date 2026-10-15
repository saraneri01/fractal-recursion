"""Task-level analysis of the final rapid HRM series (paper, Appendix A).

Compares the baseline (model A) and nested variant (model C) checkpoints at
10,000 outer steps on the 80 development tasks and writes one CSV per seed.

Usage:
    python -m eval.rapid_hrm_taskwise --seed 0
"""
from __future__ import annotations

import argparse
import random
import csv
import numpy as np
import torch

from data.rapid.arc_dataset import (
    load_arc_tasks,
    task_level_kfold,
    canvas_to_grid,
    PAD_ID,
)
from data.rapid.task_dataset import build_eval_sample

from models.rapid.hrm_baseline import (
    FlatTaskConfig,
    FlatTaskRapid,
)
from models.rapid.hrm_fractal import (
    FractalTaskConfig,
    FractalTaskRapid,
)

DATA_DIR = "datasets/arc-agi-1/data/training"

RUN_DIR = "runs/rapid_hrm_final"
CHECKPOINT_STEP = 10000


def run_paths(seed: int):
    """Checkpoint and output paths of the final rapid HRM series for one seed."""
    a_ckpt = f"{RUN_DIR}/seed{seed}/A_ckpt/step_{CHECKPOINT_STEP}.pt"
    c_ckpt = f"{RUN_DIR}/seed{seed}/C_ckpt/step_{CHECKPOINT_STEP}.pt"
    output_csv = f"{RUN_DIR}/taskwise_seed{seed}_step{CHECKPOINT_STEP}.csv"
    return a_ckpt, c_ckpt, output_csv


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLIT_SEED = 0
EVAL_SEED = 12345
N_SUP = 4
CANVAS = 30


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_and_load(model_type, path):
    seed_everything(0)

    if model_type == "A":
        model = FlatTaskRapid(
            FlatTaskConfig()
        ).to(DEVICE)
    else:
        model = FractalTaskRapid(
            FractalTaskConfig()
        ).to(DEVICE)

    ckpt = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    return model


def init_state(model_type, model):
    if model_type == "A":
        z_H, z_L = model.init_state(1, DEVICE)
        return (z_H, z_L)

    z_H, z_M, z_L = model.init_state(1, DEVICE)
    return (z_H, z_M, z_L)


def forward_segment(
    model_type,
    model,
    query_input,
    demo_inputs,
    demo_outputs,
    demo_mask,
    state,
):
    if model_type == "A":
        z_H, z_L = state

        z_H, z_L, y_hat = model(
            query_input,
            demo_inputs,
            demo_outputs,
            demo_mask,
            z_H,
            z_L,
        )

        return (z_H, z_L), y_hat

    z_H, z_M, z_L = state

    z_H, z_M, z_L, y_hat = model(
        query_input,
        demo_inputs,
        demo_outputs,
        demo_mask,
        z_H,
        z_M,
        z_L,
    )

    return (z_H, z_M, z_L), y_hat


@torch.no_grad()
def evaluate_task(model_type, model, task):

    total_real = 0
    correct_real = 0

    real_errors = 0
    pad_errors = 0
    full_canvas_errors = 0

    case_exacts = []
    case_dims = []

    for pair in task.get("test", []):

        single = {
            "train": task["train"],
            "test": [pair],
        }

        sample = build_eval_sample(single)

        query_input = sample["query_input"].to(DEVICE)
        demo_inputs = sample["demo_inputs"].to(DEVICE)
        demo_outputs = sample["demo_outputs"].to(DEVICE)
        demo_mask = sample["demo_mask"].to(DEVICE)

        target_canvas = torch.from_numpy(
            sample["query_target_canvas"]
        ).long().to(DEVICE)

        target_grid = sample["query_target_grid"]

        state = init_state(
            model_type,
            model,
        )

        for _ in range(N_SUP):
            state, y_hat = forward_segment(
                model_type,
                model,
                query_input,
                demo_inputs,
                demo_outputs,
                demo_mask,
                state,
            )

        pred = (
            y_hat.argmax(dim=-1)
            .reshape(CANVAS, CANVAS)
        )

        real_mask = target_canvas != PAD_ID
        pad_mask = target_canvas == PAD_ID

        correct_real += int(
            (
                pred[real_mask]
                == target_canvas[real_mask]
            ).sum().item()
        )

        total_real += int(
            real_mask.sum().item()
        )

        this_real_errors = int(
            (
                pred[real_mask]
                != target_canvas[real_mask]
            ).sum().item()
        )

        this_pad_errors = int(
            (
                pred[pad_mask]
                != PAD_ID
            ).sum().item()
        )

        real_errors += this_real_errors
        pad_errors += this_pad_errors
        full_canvas_errors += (
            this_real_errors
            + this_pad_errors
        )

        pred_np = pred.cpu().numpy()
        pred_grid = canvas_to_grid(pred_np)

        same_shape = (
            pred_grid.shape
            == target_grid.shape
        )

        exact = bool(
            same_shape
            and np.array_equal(
                pred_grid,
                target_grid,
            )
        )

        case_dims.append(same_shape)
        case_exacts.append(exact)

    real_acc = (
        100.0 * correct_real / total_real
        if total_real
        else 0.0
    )

    return {
        "real_acc": real_acc,
        "real_errors": real_errors,
        "pad_errors": pad_errors,
        "full_errors": full_canvas_errors,
        "correct_dims": bool(case_dims) and all(case_dims),
        "exact": bool(case_exacts) and all(case_exacts),
        "n_cases": len(case_exacts),
    }


def main():
    parser = argparse.ArgumentParser(description="Task-level comparison of the final rapid HRM baseline (A) and nested variant (C) for one seed.")
    parser.add_argument("--seed", type=int, required=True, choices=range(5), help="Training seed of the evaluated checkpoints (0-4).")
    args = parser.parse_args()
    A_CKPT, C_CKPT, OUTPUT_CSV = run_paths(args.seed)

    print("device:", DEVICE)
    print("N_SUP:", N_SUP)

    all_tasks = load_arc_tasks(DATA_DIR)
    ids = list(all_tasks.keys())

    _, val_ids = next(
        task_level_kfold(
            ids,
            k=5,
            seed=SPLIT_SEED,
        )
    )

    val_dict = {
        task_id: all_tasks[task_id]
        for task_id in val_ids
    }

    A = build_and_load(
        "A",
        A_CKPT,
    )

    C = build_and_load(
        "C",
        C_CKPT,
    )

    # Identical RNG trajectory for A.
    seed_everything(EVAL_SEED)

    results_A = {}

    for task_id, task in val_dict.items():
        results_A[task_id] = evaluate_task(
            "A",
            A,
            task,
        )

    # Reset identically for C.
    seed_everything(EVAL_SEED)

    results_C = {}

    for task_id, task in val_dict.items():
        results_C[task_id] = evaluate_task(
            "C",
            C,
            task,
        )

    rows = []

    for task_id in val_ids:

        a = results_A[task_id]
        c = results_C[task_id]

        row = {
            "task_id": task_id,
            "n_cases": a["n_cases"],

            "A_real_acc": a["real_acc"],
            "C_real_acc": c["real_acc"],
            "delta_C_minus_A": (
                c["real_acc"]
                - a["real_acc"]
            ),

            "A_real_errors": a["real_errors"],
            "C_real_errors": c["real_errors"],

            "A_pad_errors": a["pad_errors"],
            "C_pad_errors": c["pad_errors"],

            "A_full_errors": a["full_errors"],
            "C_full_errors": c["full_errors"],

            "A_correct_dims": a["correct_dims"],
            "C_correct_dims": c["correct_dims"],

            "A_exact": a["exact"],
            "C_exact": c["exact"],
        }

        rows.append(row)

    deltas = np.array([
        r["delta_C_minus_A"]
        for r in rows
    ])

    c_wins = sum(
        r["delta_C_minus_A"] > 0
        for r in rows
    )

    a_wins = sum(
        r["delta_C_minus_A"] < 0
        for r in rows
    )

    ties = len(rows) - c_wins - a_wins

    c_fewer_errors = sum(
        r["C_full_errors"]
        < r["A_full_errors"]
        for r in rows
    )

    a_fewer_errors = sum(
        r["A_full_errors"]
        < r["C_full_errors"]
        for r in rows
    )

    same_errors = len(rows) - (
        c_fewer_errors
        + a_fewer_errors
    )

    print("\n" + "=" * 72)
    print("TASK-WISE REAL-ACCURACY COMPARISON")
    print("=" * 72)

    print(
        f"C wins: {c_wins}/80"
    )

    print(
        f"A wins: {a_wins}/80"
    )

    print(
        f"ties:   {ties}/80"
    )

    print(
        f"\nmean task-level delta C-A: "
        f"{deltas.mean():+.3f} pp"
    )

    print(
        f"median task-level delta C-A: "
        f"{np.median(deltas):+.3f} pp"
    )

    print(
        f"positive-delta tasks: "
        f"{100*c_wins/len(rows):.1f}%"
    )

    print("\n" + "=" * 72)
    print("FULL-CANVAS ERROR COMPARISON")
    print("=" * 72)

    print(
        "C fewer errors:",
        c_fewer_errors,
    )

    print(
        "A fewer errors:",
        a_fewer_errors,
    )

    print(
        "same error count:",
        same_errors,
    )

    print("\n" + "=" * 72)
    print("NEAR-EXACT TASKS")
    print("=" * 72)

    for threshold in [1, 2, 5, 10, 20, 50]:

        a_n = sum(
            r["A_full_errors"] <= threshold
            for r in rows
        )

        c_n = sum(
            r["C_full_errors"] <= threshold
            for r in rows
        )

        print(
            f"<= {threshold:2d} errors: "
            f"A={a_n:2d}  C={c_n:2d}  "
            f"delta={c_n-a_n:+d}"
        )

    print("\n" + "=" * 72)
    print("DIMENSIONS")
    print("=" * 72)

    a_dims = sum(
        r["A_correct_dims"]
        for r in rows
    )

    c_dims = sum(
        r["C_correct_dims"]
        for r in rows
    )

    print(
        f"A correct dims: {a_dims}/80"
    )

    print(
        f"C correct dims: {c_dims}/80"
    )

    print("\n" + "=" * 72)
    print("TOP 10 TASKS FAVORING C")
    print("=" * 72)

    for r in sorted(
        rows,
        key=lambda x: x["delta_C_minus_A"],
        reverse=True,
    )[:10]:

        print(
            f"{r['task_id']}  "
            f"A={r['A_real_acc']:.2f}%  "
            f"C={r['C_real_acc']:.2f}%  "
            f"delta={r['delta_C_minus_A']:+.2f}  "
            f"errors A/C="
            f"{r['A_full_errors']}/"
            f"{r['C_full_errors']}"
        )

    print("\n" + "=" * 72)
    print("TOP 10 TASKS FAVORING A")
    print("=" * 72)

    for r in sorted(
        rows,
        key=lambda x: x["delta_C_minus_A"],
    )[:10]:

        print(
            f"{r['task_id']}  "
            f"A={r['A_real_acc']:.2f}%  "
            f"C={r['C_real_acc']:.2f}%  "
            f"delta={r['delta_C_minus_A']:+.2f}  "
            f"errors A/C="
            f"{r['A_full_errors']}/"
            f"{r['C_full_errors']}"
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "\nSaved:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()

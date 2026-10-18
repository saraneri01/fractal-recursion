"""Segment-wise trajectory audit of HRM baseline and HRM-F C2, seed 0 (paper, Appendix B).

Evaluates both step-4886 checkpoints with fixed weights on the same 79 batches
of the training loader, following each sequence through 16 ACT segments. For
every sequence and segment it records token accuracy, exactness (all valid
labelled positions correct), the number of changed predictions and the
relative change of each internal state. The summary counts sequences that are
exact at some segment, exact at the final segment, and that lose correctness
temporarily or through the final segment.

Usage:
    python -m scripts.audit_hrm_trajectory
"""
import os
os.environ["DISABLE_COMPILE"] = "1"

import gc
import hashlib
import itertools
import json
from pathlib import Path

import torch
import yaml

from train.hrm import PretrainConfig, create_dataloader, create_model
from models.components.hrm.losses import IGNORE_LABEL_ID

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "A": ROOT / "checkpoints/hrm/baseline_seed0",
    "C2": ROOT / "checkpoints/hrm/fractal_c2_seed0",
}
OUT = ROOT / "runs/audits/hrm_trajectory_seed0_step4886.json"
N_BATCHES = 79
STEPS = 16

def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    configs = {}
    for label, folder in RUNS.items():
        with (folder / "all_config.yaml").open() as f:
            configs[label] = PretrainConfig(**yaml.safe_load(f))
        configs[label].checkpoint_path = None
        configs[label].eval_save_outputs = []
        assert (folder / "step_4886").is_file()

    a, c = configs["A"], configs["C2"]
    assert Path(a.data_path).resolve() == Path(c.data_path).resolve()
    assert a.global_batch_size == c.global_batch_size
    assert a.seed == c.seed

    # Sample once, before constructing either model.
    torch.manual_seed(12345)
    loader, metadata = create_dataloader(
        c, "train", rank=0, world_size=1,
        test_set_mode=False, epochs_per_iter=1,
        global_batch_size=c.global_batch_size,
    )
    batches = [
        {k: v.detach().cpu().clone() for k, v in batch.items()}
        for _, batch, _ in itertools.islice(loader, N_BATCHES)
    ]
    assert len(batches) == N_BATCHES
    del loader

    digest = hashlib.sha256()
    for batch in batches:
        for key in sorted(batch):
            value = batch[key].contiguous()
            digest.update(key.encode())
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.numpy().tobytes())

    result = {
        "split": "train",
        "sampling_seed_torch": 12345,
        "dataset_seed": c.seed,
        "n_batches": N_BATCHES,
        "steps": STEPS,
        "batch_sha256": digest.hexdigest(),
        "state_change_definition": "L2(new-old)/max(L2(old),1e-12), per sequence",
        "models": {},
    }

    for label, config in configs.items():
        print(f"\n===== {label} =====", flush=True)
        torch.manual_seed(config.seed)
        model, optimizers, optimizer_lrs = create_model(
            config, metadata, world_size=1
        )
        del optimizers, optimizer_lrs

        checkpoint = RUNS[label] / "step_4886"
        state = torch.load(checkpoint, map_location="cpu")
        clean = {
            k.replace("_orig_mod.", ""): v for k, v in state.items()
        }
        assert len(clean) == len(state), "Key collision after removing the compile prefix"
        model.load_state_dict(clean, strict=True)
        del state, clean
        model.eval()
        assert model.model.config.halt_max_steps == STEPS

        rows = []
        with torch.inference_mode():
            for batch_index, cpu_batch in enumerate(batches):
                batch = {k: v.cuda() for k, v in cpu_batch.items()}
                with torch.device("cuda"):
                    carry = model.initial_carry(batch)
                mask = batch["labels"] != IGNORE_LABEL_ID
                counts = mask.sum(-1)
                valid = counts > 0

                accuracies, exacts, changes = [], [], []
                state_changes = {}
                prev_pred = None
                prev_states = None

                for step in range(1, STEPS + 1):
                    carry, _, _, outputs, finished = model(
                        carry=carry, batch=batch, return_keys=["logits"]
                    )
                    assert torch.all(carry.steps == step)
                    assert torch.all(carry.halted == (step == STEPS))
                    assert bool(finished.item()) == (step == STEPS)
                    for key in batch:
                        assert torch.equal(carry.current_data[key], batch[key])

                    pred = outputs["logits"].argmax(-1)
                    correct = ((pred == batch["labels"]) & mask).sum(-1)
                    accuracies.append(
                        (correct.float() / counts.clamp_min(1)).cpu()
                    )
                    exacts.append(((correct == counts) & valid).cpu())
                    changes.append(
                        torch.zeros_like(counts).cpu() if prev_pred is None
                        else ((pred != prev_pred) & mask).sum(-1).cpu()
                    )

                    current = {
                        name: value.detach().clone()
                        for name, value in vars(carry.inner_carry).items()
                        if torch.is_tensor(value)
                    }
                    if prev_states is not None:
                        for name, value in current.items():
                            new = value.float().flatten(1)
                            old = prev_states[name].float().flatten(1)
                            relative = (
                                (new-old).norm(dim=1)
                                / old.norm(dim=1).clamp_min(1e-12)
                            )
                            state_changes.setdefault(name, []).append(
                                relative.cpu()
                            )
                    prev_states = current
                    prev_pred = pred.clone()

                acc = torch.stack(accuracies, 1)
                exact = torch.stack(exacts, 1)
                changed = torch.stack(changes, 1)
                states = {
                    k: torch.stack(v, 1) for k, v in state_changes.items()
                }

                for index in torch.where(valid.cpu())[0].tolist():
                    rows.append({
                        "batch": batch_index,
                        "row": index,
                        "puzzle_identifier": int(
                            cpu_batch["puzzle_identifiers"][index].item()
                        ),
                        "accuracy": acc[index].tolist(),
                        "exact": exact[index].tolist(),
                        "changed_valid_tokens": changed[index].tolist(),
                        "relative_state_changes": {
                            k: v[index].tolist() for k, v in states.items()
                        },
                    })
                print(
                    f"{label}: batch {batch_index+1}/{N_BATCHES}",
                    flush=True,
                )
                del carry, outputs, batch, current, prev_states, prev_pred

        ever = sum(any(r["exact"]) for r in rows)
        lost_final = sum(
            any(r["exact"]) and not r["exact"][-1] for r in rows
        )
        lost_any = sum(
            any(x and not y for x, y in zip(r["exact"], r["exact"][1:]))
            for r in rows
        )
        summary = {
            "n_sequences": len(rows),
            "ever_exact": ever,
            "final_exact": sum(r["exact"][-1] for r in rows),
            "ever_lost_correctness": lost_any,
            "lost_correctness_by_final_step": lost_final,
            "loss_fraction_among_ever_exact": lost_any / ever if ever else None,
            "mean_accuracy_by_step": [
                sum(r["accuracy"][s] for r in rows) / len(rows)
                for s in range(STEPS)
            ],
        }
        result["models"][label] = {
            "checkpoint": str(checkpoint),
            "summary": summary,
            "sequences": rows,
        }
        print(json.dumps(summary, indent=2), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    assert [
        (r["batch"], r["row"], r["puzzle_identifier"])
        for r in result["models"]["A"]["sequences"]
    ] == [
        (r["batch"], r["row"], r["puzzle_identifier"])
        for r in result["models"]["C2"]["sequences"]
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"\nSaved: {OUT}", flush=True)

if __name__ == "__main__":
    main()

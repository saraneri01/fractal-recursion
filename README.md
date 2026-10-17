# Nested Shared-Operator Recursion on ARC-AGI-1

Code for the experiments in *Nested Shared-Operator Recursion on ARC-AGI-1:
An Empirical Study of HRM, TRM and URM*.

The repository compares baseline and nested configurations of the
Hierarchical Reasoning Model (HRM), Tiny Recursive Model (TRM) and Universal
Reasoning Model (URM). Within each model family, the nested variant preserves
the parameter count; computation, gradient propagation and input coupling are
documented separately in the paper.

## Repository layout

| Path | Content |
| --- | --- |
| `models/` | Official-codebase baselines and nested variants (`*_baseline.py`, `*_fractal*.py`) |
| `models/rapid/` | Adapted ("rapid") HRM and TRM implementations with explicit demonstration conditioning |
| `models/components/` | Layers, losses, embeddings and optimisers used by each model family |
| `configs/` | Configurations of the reported runs (`hrm/`, `trm/`, `urm/`) |
| `data/` | Dataset builders and loaders |
| `train/` | Training entry points |
| `eval/` | Evaluation entry points and ARC evaluators |
| `scripts/` | Structural, gradient and diagnostic audits |
| `results/` | Recorded outputs underlying the reported results, with provenance |
| `licenses/` | Upstream license texts |

In the rapid experiments, model `A` denotes the baseline and model `C` the
nested variant; these labels are kept because they appear in the recorded
results.

## Installation

Python 3.12 and PyTorch 2.5.1 (CUDA 12.1) were used for packaging.

```bash
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build first. The HRM and URM attention
layers import the `flash_attn_interface` module, which is provided by a
FlashAttention installation and is not installed by `requirements.txt`.
Exact package versions of individual historical runs may differ.

All commands are run from the repository root.

## 1. Data

Datasets and checkpoints are not included. Expected layout:

| Setting | Directory |
| --- | --- |
| Rapid experiments (raw ARC-AGI-1 tasks) | `datasets/arc-agi-1/data/{training,evaluation}/` |
| HRM | `datasets/hrm/arc-aug-1000/` |
| TRM | `datasets/trm/arc1concept-aug-1000/` |
| URM training | `datasets/urm/arc-aug-1000-trainonly/` |
| URM evaluation | `datasets/urm/arc-aug-1000-eval-v1/` |

**HRM and TRM.** The builders follow the upstream HRM and TRM preprocessing:
ARC-AGI-1 training and evaluation tasks plus ConceptARC, 1,000 augmentations
per task, seed 42 (the builder defaults). The commands below reconstruct the
preprocessing with the options of the packaged builders; they were not
re-executed during packaging, so compare a rebuilt dataset with
`data/expected_metadata.json`.

```bash
# HRM: ARC-AGI-1 task directory and ConceptARC corpus, as in the upstream HRM repository
python -m data.hrm.build_arc \
    --dataset-dirs <ARC-AGI>/data <ConceptARC>/corpus \
    --output-dir datasets/hrm/arc-aug-1000

# TRM: combined ARC-AGI-1 and ConceptARC JSON files, as in the upstream TRM repository
python -m data.trm.build_arc \
    --input-file-prefix <TRM data>/kaggle/combined/arc-agi \
    --output-dir datasets/trm/arc1concept-aug-1000 \
    --subsets training evaluation concept \
    --test-set-name evaluation
```

**URM.** The URM runs use the HRM-prepared data together with the TRM
`test_puzzles.json`. The links are created with:

```bash
python data/urm/prepare_links.py \
    --hrm-data datasets/hrm/arc-aug-1000 \
    --trm-data datasets/trm/arc1concept-aug-1000 \
    --output-root datasets/urm
```

`data/expected_metadata.json` records the metadata of the prepared datasets
used in the reported runs (sequence length, vocabulary, number of puzzle
identifiers and groups) and can be used to check a rebuilt dataset.

## 2. Training

Checkpoints are written to `checkpoints/<family>/<run>/`.

**HRM** (paper, Table 3):

```bash
python -m train.hrm --config-name baseline_seed0
python -m train.hrm --config-name baseline_seed1
python -m train.hrm --config-name fractal_c1_seed0
python -m train.hrm_c2_seed0
python -m train.hrm --config-name fractal_c2_seed1
```

`train.hrm_c2_seed0` reuses the C1 seed-0 configuration with the C2
architecture and stops at 4,886 optimiser updates.

**TRM** (paper, Tables 3–4), 50,000 optimiser updates:

```bash
python -m train.trm --config-name baseline_seed0_train
python -m train.trm --config-name fractal_seed0_train
```

**URM** (paper, Tables 3 and 5):

```bash
python -m train.urm --config-name baseline_seed0_60ep
python -m train.urm --config-name fractal_seed0_60ep
python -m train.urm --config-name baseline_seed0_120ep
python -m train.urm --config-name fractal_seed0_120ep
```

The 120-epoch runs are trained from scratch. Runs used only by the
diagnostics in Appendix B:

```bash
python -m train.urm --config-name fractal_8calls_seed0_30ep        # eight-application pilot
python -m train.urm --config-name fractal_damped_a025_seed0_30ep   # damping, alpha = 0.25
python -m train.urm --config-name fractal_damped_a025_seed0_60ep
python -m train.urm --config-name fractal_damped_a100_seed0_30ep   # matched control, alpha = 1.0
```

**Rapid HRM, final series** (paper, Appendix A), for each seed 0–4:

```bash
python -m train.rapid_hrm --model A --init_seed <SEED> --max_steps 10000 --eval_every 500 --n_sup 4
python -m train.rapid_hrm --model C --init_seed <SEED> --max_steps 10000 --eval_every 500 --n_sup 4
```

Four supervised segments give four optimiser updates per outer step. The
recorded baseline (model A) run for seed 0 was produced with an earlier
version of this entry point; its provenance is documented in
`results/rapid_hrm_final/`.

**Rapid TRM** (paper, Appendix A), training and evaluation in one run:

```bash
python -m train.rapid_trm --seeds 1 2 3 4
```

## 3. Evaluation

**HRM.** Save the predictions of a checkpoint, then compute original-query
and original-task scores at k:

```bash
python -m eval.hrm checkpoint=checkpoints/hrm/baseline_seed0/step_4886
python -m eval.hrm_candidates \
    --dataset datasets/hrm/arc-aug-1000 \
    --checkpoint checkpoints/hrm/baseline_seed0/step_4886 \
    --out runs/hrm_evaluations/baseline_seed0/tasks.csv
```

**TRM.** Evaluates the 50,000-update checkpoints:

```bash
python -m train.trm --config-name baseline_seed0_eval
python -m train.trm --config-name fractal_seed0_eval
```

The saved submissions are in `results/trm_official/`. ARC/pass@1 and
ARC/pass@2 can be recomputed from them on the CPU; the submissions contain
two candidates per query, so larger k cannot be recomputed this way:

```bash
python -m scripts.recompute_trm_pass12
```

**URM.** Evaluates one run on the held-out test queries (16 loops), using the
augmentation-identifier decoding in `eval/urm_arc.py`, and writes
`runs/urm_evaluations/<run>/metrics.json`:

```bash
python -m eval.urm --run baseline_seed0_60ep
python -m eval.urm --run fractal_seed0_60ep
python -m eval.urm --run baseline_seed0_120ep
python -m eval.urm --run fractal_seed0_120ep
python -m eval.urm --run fractal_damped_a025_seed0_30ep
python -m eval.urm --run fractal_damped_a025_seed0_60ep
```

With `--save-predictions`, every evaluated sequence is also saved as `.npy`
arrays (about 1 GB per run). ARC/pass@k can then be recomputed on the CPU,
without loading a model:

```bash
python -m scripts.recompute_urm_passk \
    --runs runs/urm_evaluations/baseline_seed0_60ep runs/urm_evaluations/fractal_seed0_60ep
```

The recomputation implements cropping, query matching, candidate ranking and
scoring separately from the ARC evaluator, but reuses the dataset's inverse
geometric transform.

**Rapid HRM task-level analysis** (paper, Appendix A), for each seed 0–4:

```bash
python -m eval.rapid_hrm_taskwise --seed <SEED>
```

## 4. Audits and diagnostics

Structural and gradient-path checks (paper, Section 3):

```bash
python -m scripts.audit_hrm_state_parity
python -m scripts.audit_hrm_gradpath
python -m scripts.audit_trm_structure
python -m scripts.audit_trm_gradpath
```

URM diagnostics on training-pipeline batches (paper, Appendix B):

```bash
python -m scripts.audit_urm_dynamics                                  # internal-state dynamics
python -m scripts.audit_urm_input                                     # direct input to intermediate-state updates
python -m scripts.audit_urm_6vs8                                      # eight-application pilot
python -m scripts.audit_urm_damping_sweep --alpha 0.25                # also 0.5, 0.75, 1.0
python -m scripts.audit_urm_damping_trained --comparison matched_control_30ep
python -m scripts.audit_urm_damping_trained --comparison original_vs_damped_30ep
python -m scripts.audit_urm_damping_trained --comparison original_vs_damped_60ep
```

Some diagnostics read checkpoint files named `*_compiled_audit_v1.pt` or
`*_uncompiled_audit_v1.pt`: these are copies of the corresponding training
checkpoints saved in the state-dict key format expected by the audit loader.
`eval.urm` and the URM and TRM audit scripts refuse to overwrite existing
outputs.

## Recorded results

`results/` contains the recorded outputs used in the paper, organised by
setting (`hrm_official/`, `trm_official/`, `urm_official/`,
`urm_diagnostics/`, `rapid_hrm_final/`, `rapid_trm/`). Provenance files
(`provenance.json`) describing the originating runs are included for
`hrm_official/`, `urm_official/`, `urm_diagnostics/` and
`rapid_hrm_final/logs/`.

- Metric files retain their original units; conversions to percentages
  depend on the metric and source file. Augmented-sequence exact accuracy,
  query-level correctness and entirely solved tasks are distinct measures.
- URM original-query evaluation: the HRM-prepared data used for URM encode
  augmentations in puzzle identifiers as `<task>_t<transform>_<permutation>`,
  a format not handled by the upstream URM decoder; `eval/urm_arc.py` adds
  decoding for it. The 60-epoch baseline and URM-F evaluations are in
  `results/urm_official/eval_60ep/`: for each run, `audit_summary.json`
  (metrics, effective configuration and source hashes) and the ARC submission
  in `evaluator_ARC_step_4807/`. `recompute_passk_output.txt` contains the
  ARC/pass@k recomputed from the saved predictions, with the number of solved
  queries and entirely solved tasks at each k. Saved predictions and
  checkpoints are not included because of their size.
- HRM: `results/hrm_official/wandb_summaries/` contains, for the runs whose
  evaluation metrics were recorded during training, the W&B summary and the
  run configuration; `results/hrm_official/original_task_scores/` contains the
  task-level scores at k for the seed-0 baseline and C2 checkpoints.
- ARC/pass@k is the mean over tasks of the fraction of test queries whose
  expected output is among the first k distinct ranked candidates; it differs
  from the fraction of entirely solved tasks.
- Original-query scores were checked as follows. HRM: task-level scores at k
  computed with `eval.hrm_candidates`
  (`results/hrm_official/original_task_scores/`). TRM: ARC/pass@1 and
  ARC/pass@2 recomputed from the saved submissions
  (`results/trm_official/recompute_pass12_output.txt`); values at larger k
  are those recorded by the evaluator. URM, 60 epochs: ARC/pass@k at k = 1, 2,
  100 and 1000 recomputed from the saved predictions
  (`results/urm_official/eval_60ep/recompute_passk_output.txt`).
- For the 120-epoch and damped runs, only the sequence-level metrics in
  `results/urm_official/` (token accuracy, loss and augmented-sequence exact
  accuracy) are used in the paper.
- Only the final rapid HRM series is included; the preliminary rapid HRM
  experiments mentioned in Appendix A are not packaged.
- Absolute paths in logs and provenance files refer to the original
  experiment machine.

## Attribution and licensing

This repository adapts code from the following projects:

| Project | Repository | Commit used | License |
| --- | --- | --- | --- |
| HRM | https://github.com/sapientinc/HRM | `ac15626f8db096a63c775b84c9dc868776a6feda` | Apache-2.0 (`licenses/HRM_LICENSE`) |
| TRM | https://github.com/SamsungSAILMontreal/TinyRecursiveModels | `c01103738605ba39d1430519b1ee0c62f4c707f8` | MIT (`licenses/TRM_LICENSE`) |
| URM | https://github.com/UbiquantAI/URM | `c14e55f5f9227873617015cf60a239126b55adcd` | No license specified upstream |

Files derived from these projects were modified for the experiments reported
in the paper; the listed commits identify the upstream checkouts.

Original code in this repository is released under the Apache License 2.0
(see `LICENSE`).

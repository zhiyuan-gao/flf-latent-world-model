# Endpoint latent predictor experiment

This file records the agreed first-baseline configuration. Change one item at a
time in later ablations and record the changed value with the resulting run.

## Fixed first-baseline data contract

- Tasks: `KettleBoiling`, `LoadDishwasher`, `PreSoakPan`, `RinseSinkBasin`.
- Episode split per task: 50 train / 10 validation / 20 test.
- Control frequency: 20 Hz.
- Window horizon: 16 actions, predicting only the visual endpoint at `t+16`.
- Window stride: 4 control steps (0.2 seconds); adjacent 16-step windows overlap
  by 12 actions.
- A sample is valid only when the recorded episode contains both the anchor at
  `t` and the real target at `t+16`.
- Incomplete tail windows are dropped. They are not padded, truncated, or
  assigned a repeated terminal target.
- Windows containing a dataset stage named `done` between `t` and `t+16` are
  dropped because this baseline has no terminal/absorbing-state model.
- Feature cache: frozen V-JEPA 2 ViT-G, 256 px input, native
  `16 x 16 x 1408` FP16 grids, no PCA or spatial pooling.

The committed manifest is
`outputs/checkvla_offline_predictor/manifest.json`; it currently contains
28,896 train, 5,547 validation, and 10,897 test windows.

### Generated native feature cache (2026-08-03)

- Location: `outputs/checkvla_offline_predictor/features/vjepa2_native_256`.
- Coverage: 200 train + 40 validation + 80 test trajectories (320 total).
- Cached content: 46,620 unique `t`/`t+16` frames, not every video frame.
- Tensor contract: FP16 `16 x 16 x 1408`, no projection or pooling.
- Storage: memory-mapped NPY frame/feature pairs with atomic temporary-file
  replacement; 31.3005 GiB total. Endpoint samples read only their requested
  `t` and `t+16` grids instead of materializing a complete episode cache.
- Full audit passed: all 320 files matched the manifest frame indices and
  shapes; all 16,804,085,760 feature values were finite; no temporary files
  remained.

### Why incomplete tails are dropped

This is an endpoint world-model target, not an action-policy target. If an
episode ends before `t+16`, neither the real image nor its V-JEPA latent exists,
so there is no supervised endpoint loss to compute. Repeating the last image
would silently assert an absorbing/static terminal-state model; zero padding
would create an artificial visual state. The first baseline assumes neither.

With stride 4, exactly four stride-aligned anchors near the end of each
trajectory lack a 16-step future (for trajectories longer than 16 steps). For
the 200 training trajectories this is about 800 unavailable anchors, before
the separate `done`-stage exclusion. This is small relative to the 28,896 real
training windows and does not alter the fixed-horizon objective.

This choice follows the complete-clip convention in the official V-JEPA 2-AC
DROID loader: it requires the requested clip length and samples only a full
window. In contrast, action-chunk policies such as ACT retain near-terminal
action targets by zero-padding the action suffix and supplying an `is_pad`
mask. That is appropriate because their loss is per action position. It does
not provide the missing visual target required by this endpoint predictor.

- V-JEPA 2-AC DROID loader:
  <https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_droid/droid.py>
- ACT dataset implementation:
  <https://github.com/tonyzhaozh/act/blob/main/utils.py>
- CheckVLA paper (rolling predictions stop when the episode finishes):
  <https://arxiv.org/abs/2607.26789>

## Fixed first-baseline model and loss

- Inputs: current visual grid, one current standardized proprio token, and 16
  ordered standardized action tokens.
- Predictor: width 960, depth 7, 12 attention heads, MLP ratio 4.
- Output: one residual visual prediction at `t+16`; no intermediate visual or
  future-proprio predictions and no autoregressive rollout.
- Training loss: unweighted Smooth L1 / Huber latent endpoint loss
  (`dynamic_weight=0`).
- Model-selection score: validation normalized error relative to the
  persistence baseline. Lower is better.

## Agreed training schedule

- Distributed training: two-process PyTorch DDP on `cuda:0` and `cuda:1`.
  Validation and test windows are also sharded across both ranks and their
  sufficient statistics are merged without padding or duplicate samples.
- Global batch size: 128, divided as 64 samples per GPU.
- Input pipeline: 2 persistent data workers per DDP rank (4 total), backed by
  the memory-mapped native feature cache and shared OS page cache.
- Optimizer: AdamW, learning rate `2e-4`, weight decay `0.05`.
  The CUDA run uses the fused AdamW implementation.
- Linear warmup over 5% of the selected global-step budget, followed by cosine
  decay.
- First seed (`seed=0`) pilot: 625 global optimizer steps.
- Formal upper bound: 2,250 global optimizer steps.
- Run complete validation every 125 optimizer steps.
- Early stop after three consecutive validations without a lower validation
  normalized error.
- If the pilot passes loss, memory, throughput, and action-sensitivity checks,
  train one formal run with `seed=0`. Review its complete validation and test
  results before deciding whether seeds 1 and 2 are necessary.
- Unless `--output-dir` is supplied, runs are separated as
  `seed_<seed>_steps_<budget>` so the 625-step pilot cannot overwrite or
  contaminate the 2,250-step formal run.

The step conversion preserves sample exposure exactly:

| Phase | Previous (`batch=16`) | Current (`batch=128`) | Samples |
| --- | ---: | ---: | ---: |
| Pilot | 5,000 steps | 625 steps | 80,000 |
| Formal cap | 18,000 steps | 2,250 steps | 288,000 |
| Validation interval | 1,000 steps | 125 steps | 16,000 |

The batch benchmark on both A40 GPUs measured approximately 110, 227, 327,
and 455 samples/s for global batches 16, 32, 64, and 128. Batch 128 used 5.97
GiB per GPU and retained the same throughput on the complete 200-trajectory
training split after switching the cache to memory mapping.

Pilot command after native feature extraction:

```bash
.venv-vjepa2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/train_endpoint_predictor.py \
  --seed 0 \
  --max-steps 625 \
  --skip-test
```

Launching the training script directly without `torch.distributed.run` is a
supported single-GPU fallback, not the agreed experiment command.

Two-GPU smoke verification on 2026-08-03 passed: both ranks completed forward,
backward, fused optimizer updates, checkpointing, and sharded validation. The
merged DDP validation metrics matched a single-process calculation over the
same samples.

The pilot does not evaluate the test split. Use validation diagnostics to
decide whether the configuration is healthy; reserve test evaluation for the
frozen formal configuration.

## Completed seed-0 runs (2026-08-03)

The sample-matched pilot and the formal run both used two-process DDP with
global batch 128.  Enlarging the batch preserved the model, endpoint target,
loss, window stride, and sampling weights.  It does change the optimizer's
mini-batch averaging and therefore is sample-exposure-equivalent, not an
identical update trajectory to batch 16.

| Run | Completed steps | Best step | Best val normalized error | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Pilot | 625 | 625 | 0.76680 | 2.98 min |
| Formal seed 0 | 1,875 (early stopped) | 1,500 | 0.71436 | 9.17 min |

The formal run stopped after validation failed to improve at steps 1,625,
1,750, and 1,875.  Its frozen best checkpoint was evaluated once on all 10,897
test windows:

| Metric | Overall | KettleBoiling | LoadDishwasher | PreSoakPan | RinseSinkBasin |
| --- | ---: | ---: | ---: | ---: | ---: |
| normalized error | 0.71420 | 0.72271 | 0.72209 | 0.70650 | 0.70719 |
| delta cosine | 0.52011 | 0.50793 | 0.50707 | 0.53286 | 0.53081 |
| correct beats zero action | 93.65% | 93.39% | 92.06% | 94.25% | 95.46% |
| correct beats far same-task action | 83.11% | 83.79% | 81.66% | 84.25% | 82.71% |

The action-shuffle diagnostic uses a half-group cyclic shift within each task.
The previous one-position shift was invalid as a sensitivity control for this
ordered stride-4 dataset: adjacent 16-action windows overlap by 12 actions.
This diagnostic change affects evaluation only, not training or checkpoint
selection.  The pilot's corrected full-validation score was 72.58% for correct
actions beating far same-task actions (91.87% against zero actions).

Artifacts:

- Pilot: `outputs/checkvla_offline_predictor/endpoint_runs/vjepa2_native_256/seed_0_steps_625`.
- Formal: `outputs/checkvla_offline_predictor/endpoint_runs/vjepa2_native_256/seed_0_steps_2250`.
- Frozen formal checkpoint: `seed_0_steps_2250/best.pt`.
- Formal metrics: `seed_0_steps_2250/summary.json`, `history.json`, and
  `test_metrics.json`.

### Sampling distribution used by these runs

The runs intentionally retained the pre-existing sampler: every
`(task, subtask_index)` group has equal expected probability, and windows are
uniform within each group.  There are 15 such groups, so this is not strict
four-task balancing:

| Task | Subtask groups | Expected sample share |
| --- | ---: | ---: |
| KettleBoiling | 3 | 20.00% |
| LoadDishwasher | 5 | 33.33% |
| PreSoakPan | 5 | 33.33% |
| RinseSinkBasin | 2 | 13.33% |

A future strict task-balanced run must first assign 25% to each task and then
balance subtasks inside that task.  That would be a sampler ablation and cannot
be described as the same training distribution as this completed baseline.

Run the zero-action and shuffled-action sensitivity audit on validation after
the pilot:

```bash
.venv-vjepa2/bin/python scripts/evaluate_endpoint_predictor.py \
  --checkpoint outputs/checkvla_offline_predictor/endpoint_runs/vjepa2_native_256/seed_0_steps_625/best.pt \
  --split val \
  --device cuda:0
```

The initial formal run uses the 2,250-step default and only `seed=0`:

```bash
.venv-vjepa2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/train_endpoint_predictor.py \
  --seed 0
```

## Next experiment queue (agreed 2026-08-03)

The immediate experiments remain fixed to **50 training trajectories per task**
and **window stride 4**.  Do not generate a larger dataset or a stride-2
manifest yet.  The purpose of this phase is to distinguish an optimization
problem from a supervision/objective problem before spending time on scaling.

### Motivation: temporal contraction in the completed model

On all 5,547 validation windows, the frozen batch-128 formal checkpoint was
compared with the real latent grids at `t`, `t+4`, `t+8`, `t+12`, and `t+16`.
Every prediction was nearest to `t`.  If `t` is excluded, every prediction was
nearest to `t+4`; none was nearest to `t+16`.  The model therefore moves in the
correct direction enough to beat persistence, but does not yet represent a
full 16-step transition.  It may be used for pipeline smoke tests, but not as a
frozen residual-policy or CEM guidance model.

### Experiment 1: sample-matched global-batch ablation

Run the existing endpoint-only architecture from scratch with one seed.  This
run changes only the effective batch/update count; it retains the same 50/task
episodes, stride-4 manifest, model, endpoint loss, learning rate, and existing
`(task, subtask_index)`-group-balanced sampling weights.

| Setting | Completed baseline | Batch ablation |
| --- | ---: | ---: |
| global batch | 128 | 32 |
| local batch per GPU | 64 | 16 |
| maximum optimizer steps | 2,250 | 9,000 |
| validation interval | 125 | 500 |
| maximum sample exposure | 288,000 | 288,000 |
| samples between validations | 16,000 | 16,000 |
| early-stop patience | 3 validations | 3 validations |
| peak learning rate | `2e-4` | `2e-4` |

The warmup remains 5% of the complete budget, so its duration is also matched
in samples.  Use both GPUs and do not evaluate the test split:

```bash
.venv-vjepa2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/train_endpoint_predictor.py \
  --seed 0 \
  --train-episodes-per-task 50 \
  --batch-size 32 \
  --max-steps 9000 \
  --eval-every-steps 500 \
  --skip-test \
  --output-dir outputs/checkvla_offline_predictor/endpoint_runs/vjepa2_native_256/batch32_seed0_steps9000
```

Checkpoint selection continues to use complete-validation normalized endpoint
error.  After selection, run these additional diagnostics on the complete
validation split:

1. nearest real time among `t`, `t+4`, `t+8`, `t+12`, and `t+16`, overall and
   per task;
2. correct action versus zero action and far same-task action;
3. normalized endpoint error and delta cosine, overall and per task.

Interpretation is based on temporal retrieval, not endpoint MSE alone.  If
`t`/`t+4` still dominate, or endpoint MSE improves without later temporal
retrieval, reducing the batch has not solved the residual-guidance failure.  If
`t+16` becomes the modal future and the current-state collapse disappears, use
the batch result to choose the optimization setting for Experiment 2.

#### Experiment 1 result (completed 2026-08-03)

The batch-32 run early-stopped at step 4,000 after three non-improving complete
validations; its best checkpoint was step 2,500.  It completed in 6.48 minutes
and used 2.53 GiB peak allocated CUDA memory per rank.  Both models below were
reevaluated on the same 5,547 validation windows with the corrected far
same-task action control:

| Validation metric | Batch 128 baseline | Batch 32 ablation |
| --- | ---: | ---: |
| normalized endpoint error | 0.71436 | **0.71223** |
| delta cosine | 0.51839 | **0.52106** |
| correct beats zero action | **93.37%** | 89.49% |
| correct beats far same-task action | 82.64% | **83.85%** |
| prediction MSE from current `t` | 0.96720 | 1.03193 |
| nearest real time is `t` | 100.00% | 100.00% |
| nearest future time is `t+4` | 100.00% | 100.00% |
| nearest real time is `t+16` | 0.00% | 0.00% |

Reducing the batch therefore changed optimization and produced a small endpoint
improvement, but it did not change the temporal failure on any task.  Experiment
1 fails the residual-guidance gate.  Do not continue the endpoint-only model at
batch 32 or run another batch size; proceed to Experiment 2 at 50/task and
stride 4.

Artifacts:

- `outputs/checkvla_offline_predictor/endpoint_runs/vjepa2_native_256/batch32_seed0_steps9000/best.pt`
- `batch32_seed0_steps9000/summary.json`
- `batch32_seed0_steps9000/val_metrics_far_shuffle.json`
- `batch32_seed0_steps9000/val_temporal_retrieval.json`

### Experiment 2: causal multi-time supervision at the same data scale

Because Experiment 1 did not remove temporal contraction, keep 50/task and
stride 4 and replace endpoint-only supervision with shared causal predictions
at `t+4`, `t+8`, `t+12`, and `t+16`.  Every example retains the same tensor
shape and the same 16 ordered action slots.  For horizon `h`, slots after `h`
are zeroed and excluded as keys/values with a Transformer padding mask; a
learned horizon token identifies which endpoint is requested.  Thus the
`t+4` prediction cannot depend on actions 5--16, while the `t+16` prediction
uses all 16 actions.  All four horizons share one predictor and one output
projection; there are not four separate prediction heads.

The implementation has an explicit autograd regression test: loss on the
`t+4` output gives exactly zero gradient to actions 5--16.  This avoids both
future-action leakage and variable input dimensions.  The downstream planner
still uses `t+16`; earlier targets are auxiliary supervision for learning
temporal progression.

Keep the existing sampler during this architecture gate so the comparison with
the endpoint baseline changes only the predictor/objective.  The current cache
already contains every required intermediate target for all 28,896 training
windows (zero missing windows), so no feature extraction is required for this
50/task experiment.

Use the sample-matched batch-32 schedule and both GPUs:

```bash
.venv-vjepa2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/train_endpoint_predictor.py \
  --multi-horizon \
  --seed 0 \
  --train-episodes-per-task 50 \
  --batch-size 32 \
  --max-steps 9000 \
  --eval-every-steps 500 \
  --early-stop-patience 3 \
  --skip-test \
  --output-dir outputs/checkvla_offline_predictor/endpoint_runs/vjepa2_native_256/causal_multihorizon_b32_seed0_steps9000
```

Training loss is the equal-weight mean of four Huber endpoint losses.  Model
selection uses the complete-validation normalized error of `t+16`, preserving
comparability with Experiment 1.  Validation also records separate metrics for
all four horizons.  The run has a 9,000-step upper bound and may early-stop
after three consecutive 500-step validations without a lower `t+16`
normalized error.

Do not start residual-policy training merely because multi-time validation MSE
is lower.  The model must first show later-time retrieval, per-task improvement
over persistence, and correct-action sensitivity.  Local residual-action
ranking will then require a separate perturbed-action validation set.

#### Experiment 2 result (completed 2026-08-03)

The two-GPU run early-stopped at step 5,000 after three non-improving complete
validations.  Its best checkpoint was step 3,500.  It completed in 21.92
minutes and used 6.40 GiB peak allocated CUDA memory per rank.  The locked test
split was not read.

On all 5,547 validation windows, the best checkpoint produced:

| Metric | Endpoint-only batch 32 | Causal multi-time |
| --- | ---: | ---: |
| t+16 normalized error | 0.71223 | **0.70265** |
| t+16 delta cosine | 0.52106 | **0.52964** |
| correct beats zero action | **89.49%** | 87.89% |
| correct beats far same-task action | 83.85% | **85.49%** |
| nearest real time is current `t` | 100.00% | 100.00% |
| nearest future is `t+4` | 100.00% | 91.27% |
| nearest future is `t+8` | 0.00% | 5.77% |
| nearest future is `t+12` | 0.00% | 1.93% |
| nearest future is `t+16` | 0.00% | 1.03% |

All four supervised endpoints beat persistence, so none of the auxiliary
targets failed to train:

| Requested endpoint | Normalized error | Delta cosine |
| --- | ---: | ---: |
| `t+4` | 0.75355 | 0.47090 |
| `t+8` | 0.72596 | 0.50226 |
| `t+12` | 0.71207 | 0.51871 |
| `t+16` | 0.70265 | 0.52964 |

The auxiliary supervision therefore improves endpoint regression and begins
to move the future-only retrieval distribution beyond `t+4`.  It does **not**
remove the primary temporal contraction: every `t+16` prediction remains
closer to the current latent grid than to any observed future grid, for every
task.  Experiment 2 fails the residual-guidance gate.  Do not yet freeze this
checkpoint for residual-policy training or CEM, and do not scale the same
objective to more trajectories until the target/loss or transition structure
is changed and revalidated.

Artifacts:

- `outputs/checkvla_offline_predictor/endpoint_runs/vjepa2_native_256/causal_multihorizon_b32_seed0_steps9000/best.pt`
- `causal_multihorizon_b32_seed0_steps9000/summary.json`
- `causal_multihorizon_b32_seed0_steps9000/history.json`
- `causal_multihorizon_b32_seed0_steps9000/val_metrics_action_controls.json`

### Experiment 3: restore V-JEPA2-AC representation normalization and L1

The Experiment 2 checkpoint was audited before changing capacity or data
scale.  Its `t+16` predicted delta had 56.55% of the true delta RMS but only
30.86% projection onto the true future direction.  A random 2,048-window train
subset also retrieved the current state for 100% of predictions, so the
contraction is not primarily a validation-generalization failure.

The cached encoder tokens have mean per-token standard deviation 2.78 because
the encoder's learned affine LayerNorm is part of the frozen representation.
The original V-JEPA2-AC DROID training path applies another affine-free
`F.layer_norm` to both target-encoder representations and predictor outputs
before its `loss_exp=1` L1 objective.  Experiment 2 omitted this second
normalization and applied Huber directly at the cached scale.  In a 1,024
validation-window audit, dimensions with absolute `t+16` change greater than 1
were 32.51% of all dimensions but carried 95.40% of squared future-change
energy, placing most motion energy in Huber's linear region.

Experiment 3 changes only this representation/loss contract:

- apply affine-free per-token LayerNorm to cached current and target grids at
  load time; no feature cache regeneration;
- apply the same LayerNorm to each predictor output;
- optimize the equal-weight mean of the four L1 endpoint losses;
- retain 50/task, stride 4, the same windows and sampler, fixed 16 action
  slots, causal prefix masks, model capacity, batch size, and schedule.

```bash
.venv-vjepa2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/train_endpoint_predictor.py \
  --multi-horizon \
  --ac-normalized-l1 \
  --seed 0 \
  --train-episodes-per-task 50 \
  --batch-size 32 \
  --max-steps 9000 \
  --eval-every-steps 500 \
  --early-stop-patience 3 \
  --skip-test \
  --output-dir outputs/checkvla_offline_predictor/endpoint_runs/vjepa2_native_256/causal_multihorizon_acnorm_l1_b32_seed0_steps9000
```

This is a representation-contract ablation, not yet a rolling world model.
It passes the gate only if `t+16` predictions stop being universally nearest
to current `t` while retaining correct-action sensitivity.  If contraction
persists, proceed to a stepwise transition/nominal-proprio rollout rather than
scaling this direct endpoint objective.

#### Experiment 3 result (completed 2026-08-03)

The two-GPU run early-stopped at step 4,500; the best normalized-MSE checkpoint
was step 3,000.  It completed in 19.88 minutes and used 6.40 GiB peak allocated
CUDA memory per rank.  The locked test split was not read.  Runtime audits
confirmed affine-free token means below `2e-8` in absolute value and token
standard deviations within `[0.9999991, 0.9999996]` before training.

On all 5,547 validation windows, the best checkpoint produced:

| Metric in AC-normalized space | Result |
| --- | ---: |
| t+16 normalized error | 0.79153 |
| t+16 delta cosine | 0.46602 |
| correct beats zero action | 91.24% |
| correct beats far same-task action | 85.40% |
| nearest real time is current `t` | 100.00% |
| nearest future is `t+4` | 89.80% |
| nearest future is `t+8` | 6.72% |
| nearest future is `t+12` | 2.43% |
| nearest future is `t+16` | 1.05% |

The normalized error is measured after changing representation geometry and
must not be numerically compared as if it were the same raw-latent MSE as
Experiment 2.  The invariant temporal gate is unambiguous: all predictions
remain nearest to current `t`, both overall and within every task.

The normalization increased the `t+16` predicted-delta RMS ratio from 56.55%
to 63.33%, but its projection onto the true future direction stayed essentially
unchanged (30.86% before versus 30.48% after).  It therefore added predicted
motion without adding useful forward progress.  Action sensitivity improved,
but future endpoint retrieval did not: `t+16` retrieval was 1.03% before and
1.05% after.

Experiment 3 fails the residual-guidance gate.  The affine-free target/output
normalization is the correct V-JEPA2-AC representation contract and should be
retained, but it is not the cause of the contraction.  Do not scale this direct
endpoint model to more trajectories.  The next architecture experiment should
replace independent endpoints with a stepwise transition/rolling predictor
and provide a candidate-conditioned nominal proprioceptive rollout (or a
learned equivalent) without exposing recorded future proprioception.

Artifacts:

- `outputs/checkvla_offline_predictor/endpoint_runs/vjepa2_native_256/causal_multihorizon_acnorm_l1_b32_seed0_steps9000/best.pt`
- `causal_multihorizon_acnorm_l1_b32_seed0_steps9000/summary.json`
- `causal_multihorizon_acnorm_l1_b32_seed0_steps9000/history.json`
- `causal_multihorizon_acnorm_l1_b32_seed0_steps9000/val_metrics_action_controls.json`

### Later experiments, only after the 50/task gate

1. Build nested `50 -> 100 -> 200 -> 400` train-episode splits while keeping
   validation and test episodes fixed.  Retrain the selected architecture with
   four tasks sampled at 25% each and subtasks balanced inside each task.
2. Read the locked test split only after architecture, sampling, and acceptance
   criteria are frozen.
3. Compare stride 4 with stride 2 last, using the same number of sampled windows
   and optimizer sample exposure.  Stride 2 adds highly overlapping anchors;
   it does not add independent trajectories or shorten the 16-step horizon.

The datasets contain 501 episodes for the first three tasks and 509 for
`RinseSinkBasin`.  Retaining 10 validation and 20 test episodes leaves a common
maximum of 471 train episodes per task, so 400/task is the clean common formal
scale; `500 train/task` would consume held-out episodes and is not an allowed
comparison.

## Deliberately adjustable after the first baseline

- Sampling mixture across task, trajectory, window, and subtask.
- Window stride (first comparison if needed: 4 versus 2).
- Endpoint-only versus rolling/multi-time prediction.
- Predictor capacity.
- Dynamic-token loss weighting.
- Offset-1/2/3 validation windows for checking stride-phase robustness.

Do not change these during the sample-matched 625-step pilot unless the run is invalid. The
completed baseline used the existing `(task, subtask)`-group-balanced sampler;
use a separate run name if the sampling mixture is changed.

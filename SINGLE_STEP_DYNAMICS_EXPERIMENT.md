# Single-step V-JEPA2 dynamics gate

## Objective

The next experiment isolates the first requirement of the latent world model:
given the current visual state, current robot state, and the next four ordered
low-level actions, the model must predict the actual state at `t+4` instead of
remaining near the current state `t`.

This gate deliberately does **not** test direct `t+16` endpoint prediction,
two-step self-rollout, a larger predictor, more trajectories, or a residual
policy.  None of those experiments should proceed until the single learned
transition advances time on both a fixed train diagnostic set and held-out
validation episodes.

## Frozen data contract

- Tasks: `PreSoakPan`, `KettleBoiling`, `LoadDishwasher`, and
  `RinseSinkBasin`.
- Episode split: 50 train / 10 validation / 20 locked test episodes per task.
- Camera: `robot0_agentview_left`.
- Window-start stride: 4 control steps.
- Control frequency: 20 Hz; one model step therefore spans 0.2 seconds.
- Existing `t+16` manifest records remain the source of episode membership and
  window anchors.  Each record is cropped to its first four actions and the
  targets at `current_frame + 4`.  This preserves the exact old comparison set
  instead of silently adding near-terminal windows.
- Windows without cached `t` and `t+4` features are rejected during dataset
  construction.  The native cache is expected to cover every retained record.
- The locked test split is not read during training or model selection.

The V-JEPA2 feature cache is reused without re-extraction:

```text
outputs/checkvla_offline_predictor/features/vjepa2_native_256/
```

Each visual state is the frozen native V-JEPA2 grid
`[16, 16, 1408]`.  The encoder remains frozen and is not loaded by the
predictor trainer.

**Retrospective correction (2026-08-04):** the old cache directory name was
misleading.  Its metadata showed that the features were extracted with
`facebook/vjepa2-vitg-fpc64-384` weights while forcing a 256 crop.  Results
below that use this old cache remain as historical diagnostics, but they are
not native-256 V-JEPA2 results and must not be used as the main comparison.

## One-step sample

The model receives only information available before the four-action block is
executed:

\[
(z_t, s_t, a_t,a_{t+1},a_{t+2},a_{t+3})
\longrightarrow
(\hat z_{t+4},\hat s_{t+4}).
\]

- `z_t`: current frozen visual latent.
- `s_t`: current 16-D PandaOmron proprioception.
- `a[t:t+4]`: four ordered 12-D action vectors.
- `z[t+4]`: primary visual target.
- `s[t+4]`: target-only proprioceptive supervision.  It is never an input.

The 16-D state contains base position/quaternion, base-relative end-effector
position/quaternion, and two gripper coordinates.  It contains no object state
and no velocity.  Objects and contact therefore still have to be represented
by `z_t`.

There is no `t-4` visual input, no repeated anchor-state sequence, and no
recorded future proprioception in the conditioning path.

## Tokenization and predictor

- Predictor width / depth / heads: `960 / 7 / 12` (the CheckVLA-scale model).
- Visual tokens: 256 patch tokens, each mapped `1408 -> 960`.
- State token: the standardized 16-D `s_t` is mapped by one `Linear(16, 960)`.
- Action tokens: every standardized 12-D action is independently mapped by
  `Linear(12, 960)` and receives a learned within-block position embedding.
- The four actions stay as four ordered tokens.  They are never concatenated
  or averaged into one token.
- Action and state channels are standardized with train-only statistics.  No
  second vector-wise LayerNorm is applied to either condition.
- The visual and state outputs are direct next-state predictions.  In
  particular, the visual head is not zero-initialized as a persistence
  residual predictor.

The predictor returns both the visual grid and the next standardized state.
The state head is small; it is an internal rollout state rather than a target
for the generated subtask video.

## Loss

The fixed first-run objective is

\[
\mathcal L = \mathcal L_z + 0.005\,\mathcal L_s.
\]

Visual loss:

\[
\mathcal L_z =
\operatorname{mean\,Huber}(\hat z_{t+4},z_{t+4}).
\]

State loss is the equal mean of five semantic terms:

1. standardized base-position Huber;
2. sign-invariant base-quaternion distance
   `1 - abs(dot(normalize(q_hat), normalize(q)))`;
3. standardized base-relative end-effector-position Huber;
4. the same sign-invariant end-effector-quaternion distance;
5. standardized gripper Huber.

The initial proposed state weight was `0.1`.  Before the formal run, a
no-update audit measured the shared-trunk gradient norms from `L_z` and
`w_s L_s`.  The desired auxiliary-to-visual ratio is 10--30%.  The empirical
audit below rejected `0.1` and fixed `w_s=0.005` for the formal run.  There is
no adaptive loss weighting during training.

### Completed gradient audit

| Setting | Visual grad norm | State grad norm | Weighted state / visual |
| --- | ---: | ---: | ---: |
| `w_s=0.1`, audit batch 2 | 0.5614 | 21.5495 | 3.839 |
| `w_s=0.005`, audit batch 4 | 0.5426 | 17.9362 | 0.165 |

Although the unweighted scalar losses had similar scale (`L_z=1.2165`,
`L_s=0.6438` in the first audit), the state objective produced a roughly
38-times larger shared-trunk gradient.  A scalar-loss comparison would
therefore have selected a misleading weight.  The fixed `0.005` value places
the auxiliary gradient inside the intended 10--30% interval while leaving
visual dynamics primary.  Both audit JSON files are retained under
`outputs/single_step_dynamics/`.

Checkpoint selection and early stopping use visual validation quality only.
State metrics are diagnostics and a prerequisite for later autoregressive
rollout, not the model-selection objective.

## Sampling and optimization

- Four tasks receive equal probability (25% each).
- Within each task, `(task, subtask_index)` groups receive equal probability.
- Seed: 0.
- Distributed training: two-process DDP on the two A40 GPUs.
- Global batch: 32 (16 per GPU).
- Optimizer: AdamW, learning rate `3e-4`, weight decay `0.05`.
- Gradient clipping: 1.0.
- Schedule: 5% linear warmup followed by cosine decay.
- Validation: once per equivalent epoch.
- Early stopping is disabled through epoch 10.
- After epoch 10, stop after five complete validation epochs without a lower
  visual normalized error.
- Maximum: 20 equivalent epochs.
- Save an epoch checkpoint at every validation, plus `best.pt` and `last.pt`.
- Run only seed 0.  Do not evaluate the locked test split yet.

Steps per epoch are computed after loading the retained single-step windows as
`ceil(num_train_windows / global_batch)` and written to the run summary.

## Train and validation diagnostics

Every epoch evaluates:

1. the complete held-out validation split;
2. a fixed, task-balanced subset of up to 2,048 train windows.

For both sets, report overall and per-task:

- visual MSE and Huber loss;
- persistence MSE from copying `z_t`;
- normalized visual error `MSE(pred, z[t+4]) / MSE(z[t], z[t+4])`;
- fraction for which the prediction is closer to `z[t+4]` than to `z[t]`;
- normalized signed temporal margin;
- predicted/true visual-delta RMS ratio;
- visual-delta cosine;
- correct-action error versus zero-action error;
- correct-action error versus far same-task shuffled-action error;
- the analogous normalized-error and future-closer diagnostics for
  proprioception.

The "visually changed" subset is fixed before optimization as the upper half
of the task-balanced train diagnostic windows by persistence MSE.  Its median
threshold is reused unchanged for train and validation reporting.

The central diagnostic is

\[
P\left[d(\hat z_{t+4},z_{t+4}) < d(\hat z_{t+4},z_t)\right].
\]

Interpretation:

| Fixed train diagnostic | Validation | Decision |
| --- | --- | --- |
| still nearest to `t` | still nearest to `t` | stop; change the dynamics objective, not capacity/data/rollout |
| nearest to `t+4` | nearest to `t` | optimization can fit dynamics; investigate generalization and data scale |
| nearest to `t+4` | nearest to `t+4` | pass the one-step gate and prepare two-step self-rollout |
| visual loss improves but actions do not matter | either | add action-counterfactual data/conditioning; do not plan with the model |

The ordinary visual Huber loss is tested first to keep the architecture gate
comparable to published rolling predictors.  If the model still copies `t` on
the fixed train diagnostic, the next loss ablation may add an explicit
future-vs-current ranking term.  That term is not part of this first run.

## Deferred second stage

Only after the one-step gate passes, warm-start the same shared transition and
train a two-step rollout:

\[
(\hat z_{t+4},\hat s_{t+4}) = F(z_t,s_t,a_{t:t+3}),
\]

\[
(\hat z_{t+8},\hat s_{t+8}) =
F(\hat z_{t+4},\hat s_{t+4},a_{t+4:t+7}).
\]

The second-step weight will be ramped from zero.  Four-step open-loop rollout
to `t+16` remains an evaluation/planning horizon; no separate direct `t+16`
head will be restored.

## Commands

Run the no-update loss/gradient audit first:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-robocasa-gr00t/bin/python \
  scripts/train_single_step_dynamics.py \
  --device cuda:0 \
  --audit-only \
  --output-dir outputs/single_step_dynamics/audit_seed_0
```

After confirming that the weighted state-to-visual shared-gradient ratio is in
the intended range, launch the one formal seed on both GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-robocasa-gr00t/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_single_step_dynamics.py \
  --seed 0 \
  --batch-size 32 \
  --proprio-weight 0.005 \
  --min-epochs 10 \
  --max-epochs 20 \
  --early-stop-patience 5 \
  --output-dir outputs/single_step_dynamics/formal_seed_0
```

Neither command reads the locked test split.

## Completed formal run

The seed-0 formal run completed on both A40 GPUs on 2026-08-03.  No cached
features were missing, and the locked test split remained unread.

| Run property | Result |
| --- | ---: |
| Predictor parameters | 80,508,880 |
| Train / validation windows | 28,896 / 5,547 |
| Global batch / steps per epoch | 32 / 903 |
| Completed epochs / optimizer steps | 15 / 13,545 |
| Stop reason | early stop after five post-epoch-10 stale validations |
| Best checkpoint | epoch 4 |
| Elapsed wall time | 23.83 minutes |
| Peak CUDA memory on rank 0 | 2.49 GiB |

The run-level gradient audit measured a weighted state-to-visual shared-trunk
gradient ratio of `0.174`, consistent with the pre-run calibration target.

### Learning trajectory

| Epoch | Train visual normalized error | Train future-closer | Validation visual normalized error | Validation future-closer |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8709 | 0.00% | 0.8740 | 0.00% |
| **4 (best)** | **0.7417** | **0.00%** | **0.7657** | **0.00%** |
| 10 | 0.5819 | 21.68% | 0.8057 | 0.00% |
| 15 | 0.4282 | 70.07% | 0.8663 | 0.00% |

The network can fit the requested transition: by the final epoch, 70.07% of
the fixed train predictions are closer to `t+4` than to `t`.  This behavior
does not transfer to held-out episodes.  The widening train/validation gap and
the validation minimum at epoch 4 are direct evidence of overfitting under the
50-trajectory-per-task setting.

### Best-checkpoint validation

| Metric | Epoch-4 result |
| --- | ---: |
| Visual normalized error | 0.7657 |
| Dynamic-subset visual normalized error | 0.7213 |
| Prediction closer to `t+4` than `t` | 0.00% |
| Dynamic-subset prediction closer to `t+4` | 0.00% |
| Visual delta cosine | 0.4557 |
| Predicted / true visual-delta RMS | 0.5359 |
| Correct action beats zero action | 87.22% |
| Correct action beats same-task shuffled action | 77.75% |
| Proprio normalized error | 22.46 |
| Proprio prediction closer to `s[t+4]` than `s[t]` | 54.34% |

The visual normalized error below one means the predictor reduces average MSE
relative to copying `z_t`; it does **not** mean that it reaches the future
state.  Its visual delta has a partially correct direction, but its projected
advance is insufficient to cross the midpoint between `t` and `t+4` for any
validation window.  The zero/shuffle controls show that the predictor uses the
action tokens, so action neglect is not the primary failure mode.

The proprio head also fails the rollout-quality gate.  Its semantic physical-
unit loss is small, but its error is 22.46 times the very small four-step
persistence motion in standardized state space.  A 54.34% pairwise
future-closer rate does not compensate for that excessive magnitude error.
`s[t+4]` predictions therefore must not yet be fed back autoregressively.

Per-task validation visual normalized errors are similar:

| Task | Visual normalized error | Dynamic normalized error | Correct beats shuffle |
| --- | ---: | ---: | ---: |
| KettleBoiling | 0.7640 | 0.7192 | 79.49% |
| LoadDishwasher | 0.7713 | 0.7183 | 75.25% |
| PreSoakPan | 0.7639 | 0.7208 | 79.33% |
| RinseSinkBasin | 0.7607 | 0.7295 | 77.27% |

The failure is therefore shared across tasks rather than being caused by one
outlier task.

### Gate decision

This run **does not pass** the single-step dynamics gate.  Do not start the
deferred two-step self-rollout and do not use this checkpoint for CEM or
residual-action target generation.  The result places the experiment in the
second row of the decision table: the model can fit train dynamics, while
held-out predictions remain nearest to `t`.

The next controlled experiment should keep this architecture, horizon, loss,
stride, validation split, and diagnostics fixed while increasing the number
and diversity of training trajectories.  This isolates data-scale
generalization from architecture changes.  An explicit future-vs-current
ranking loss is a later ablation if additional data does not make validation
predictions advance.  Two-step rollout remains deferred until both visual and
proprio one-step gates pass.

The numerical results above are the retained experiment record.  The
`formal_seed_0` checkpoints, history, and summary artifacts were permanently
deleted after review to reclaim storage; no checkpoint from this run remains.

## 100-trajectory, stride-1 manifest and superseded cache

The next cache is ready under:

```text
outputs/single_step_dynamics/data_100_per_task_stride1/
```

It is a nested expansion of the completed 50/task run, not a fresh call that
shifts the split boundaries:

- the original 50 train episodes/task are retained;
- 50 previously unused episodes/task are added to train;
- the original 10 validation and 20 locked test episodes/task are unchanged;
- train, validation, and test episode lists are disjoint.

Window-start stride is now 1 control step for train and validation.  The source
manifest horizon remains 16 so terminal filtering is unchanged, while the
single-step loader still consumes only four actions and targets at `t+4`.
This run changes both trajectory count (50 to 100/task) and window density
(stride 4 to 1), so it is not a pure data-scale ablation.

| Cache property | Result |
| --- | ---: |
| Train / validation / locked-test episodes | 400 / 40 / 80 |
| Train / validation / locked-test manifest windows | 227,408 / 22,139 / 43,469 |
| Cached train+validation episode feature pairs | 440 |
| Cached locked-test feature pairs | 0 |
| Superseded feature storage | 172.35 GiB, deleted 2026-08-04 |
| Missing train / validation `t,t+4` features | 0 / 0 |
| Feature representation | float16 `[16,16,1408]` |
| Steps/equivalent epoch at global batch 128 | 1,777 |

The manifest, action/proprio statistics, and split episode caches remain
valid.  The original train/validation visual cache and its machine-readable
audit were generated from the mismatched 384-weight/256-crop encoder.  The
173 GiB visual cache was permanently deleted on 2026-08-04; locked-test
features were never cached.

The earlier 50/task and 100/task stride-4 feature caches were permanently
deleted because the dense stride-1 cache supersedes their frame coverage.

## Stopped 100-trajectory, stride-1 formal run (superseded encoder)

Seed 0 was launched on both A40 GPUs on 2026-08-03.  A preceding 20-update
global-batch-128 smoke run completed successfully: rank-0 peak allocated CUDA
memory was 5.84 GiB, both ranks performed optimization, and the weighted
proprio-to-visual shared-trunk gradient ratio was 0.119 (inside the calibrated
0.10--0.30 range).  The smoke run saved no checkpoint and did not read test.

The formal run is fixed to the following protocol:

| Setting | Value |
| --- | ---: |
| Train / validation windows | 227,408 / 22,139 |
| Global / per-GPU batch | 128 / 64 |
| GPUs | 2 |
| Epochs / optimizer updates | 25 / 44,425 |
| Learning rate | `3e-4` |
| Schedule | 5% warmup, cosine decay |
| Weight decay | `0.05` |
| Proprio loss weight | `0.005` |
| Early stopping | disabled |
| Full validation | every epoch |
| Saved checkpoints | epochs 5, 10, 15, 20, 25 only |
| Test | locked and uncached |

Run directory:

```text
outputs/single_step_dynamics/formal_100_per_task_stride1_b128_seed_0/
```

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv-robocasa-gr00t/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_single_step_dynamics.py \
  --manifest-dir outputs/single_step_dynamics/data_100_per_task_stride1 \
  --seed 0 --batch-size 128 \
  --learning-rate 3e-4 --weight-decay 0.05 --warmup-ratio 0.05 \
  --proprio-weight 0.005 --max-epochs 25 --min-epochs 25 \
  --disable-early-stopping --checkpoint-epochs 5 10 15 20 25 \
  --output-dir \
    outputs/single_step_dynamics/formal_100_per_task_stride1_b128_seed_0
```

This experiment intentionally answers whether substantially denser and more
diverse supervision can make held-out predictions advance toward `t+4`; it
does not isolate trajectory count, stride, batch size, or training duration as
a single-variable ablation.

The run was manually stopped after epoch 16 (28,432 optimizer updates) on
2026-08-04, once the encoder mismatch was confirmed.  At that point train
future-closer had risen to 51.27% (72.20% on the dynamic subset), while
validation future-closer remained 0.26% (0.53% dynamic).  Validation normalized
error had worsened from its epoch-4 minimum of 0.7337 to 0.8240.  These numbers
show train-set fitting and held-out failure for the superseded encoder, but the
remaining nine epochs were not worth running.

## Native-256 corrective cache

The corrected frozen encoder is:

- model: `facebook/vjepa2-vitg-fpc64-256`;
- fixed revision: `875c192b7b704b87d1e1d99345769632dd5f739a`;
- model SHA-256:
  `f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b`;
- checkpoint config: `crop_size=256`, `image_size=256`, patch size 16;
- preprocessing: bilinear short-edge resize to 292 followed by center crop 256;
- output: float16 `[16,16,1408]` per cached frame.

The encoder loader now rejects checkpoint/crop resolution mismatches.  The
unchanged 100/task stride-1 manifest was reused, and all train/validation visual
features were re-extracted under
`outputs/single_step_dynamics/data_100_per_task_stride1/features/vjepa2_native_256/`.
The two-GPU extraction completed in about 35 minutes: 440 episode pairs,
256,587 cached frame rows, 172.27 GiB, and no temporary files.  Both task-pair
metadata files identify the native-256 checkpoint.  The full loader audit found
227,408 train and 22,139 validation windows with zero missing `t,t+4` features.
The locked test split remains uncached and unread.  No native-256 predictor
training had started at the time of cache completion.

## Native-256 formal run

Seed 0 was launched from scratch on both A40 GPUs on 2026-08-04.  It does not
load or resume any checkpoint from the superseded 384-weight run.  The data,
sampling, predictor, optimizer, and evaluation protocol remain unchanged:
100 train trajectories/task, stride 1, global batch 128 (64/GPU), 25 fixed
epochs, 44,425 optimizer updates, learning rate `3e-4`, 5% warmup plus cosine
decay, weight decay `0.05`, proprio loss weight `0.005`, and full validation
after every epoch.  Checkpoints are restricted to epochs 5, 10, 15, 20, and
25; the locked test split is not read.

The only intended experimental change is the frozen visual encoder/cache:
`facebook/vjepa2-vitg-fpc64-256` with native 256 preprocessing and the
corresponding `vjepa2_native_256` cache.  The run directory and live log are:

```text
outputs/single_step_dynamics/formal_native256_100_per_task_stride1_b128_seed_0/
outputs/single_step_dynamics/formal_native256_100_per_task_stride1_b128_seed_0/train.log
```

Initial health checks passed: both distributed ranks entered optimization,
the predictor has 80.51M parameters, the shared-trunk proprio-to-visual
gradient ratio is 0.146, and each GPU uses about 7.1 GiB while training.  The
run is detached in tmux session `vjepa2_native256_train`.  Based on the
superseded run's throughput, 25 epochs are expected to take approximately
2 hours 50 minutes.

On 2026-08-04, after the native-256 run was confirmed healthy, 114.764 GiB of
superseded or unrelated artifacts were permanently removed.  This included
the stopped JEPA-WMs reproduction payload and environment, its old RoboCasa
asset checkout, the 384 V-JEPA2 checkpoint, the CheckVLA and dynamics-bakeoff
run artifacts, the superseded 384-feature single-step checkpoints, and the
separate `goal_video_model` tree.  Historical numerical records remain in the
research documents, but those deleted checkpoints and payloads are not
recoverable locally.  The active native-256 cache, native-256 checkpoint,
RoboCasa365 source data, active environment, and current formal run were not
removed.  Home usage after cleanup was 730.445 GiB, leaving approximately
219.555 GiB under the stated 950 GiB allocation.

## Queued visual-only target ablation

A controlled no-future-proprio-target ablation was implemented and queued on
2026-08-04.  It **retains the current standardized proprioceptive state
`s_t` as one input token**, because removing that condition at the same time
would confound two changes.  It removes the `s_{t+4}` output head, never moves
the recorded future state target to the GPU during training, and optimizes
only the visual Huber objective for `z_{t+4}`.  Visual evaluation, action
controls, and future-closer diagnostics remain unchanged; state prediction
metrics are absent rather than being reported from an untrained dummy head.

All shared model parameters have the same seed-0 initialization as the joint
visual/proprio run.  The optional state head is constructed and initialized
before being dropped so its RNG consumption cannot shift the shared trunk's
initialization.  Every other setting is held fixed: native-256 cache, the same
manifest and weighted sampler, 100 train trajectories/task, stride 1, global
batch 128 on two GPUs, predictor width/depth/heads `960×7×12`, 25 epochs,
44,425 updates, `3e-4` learning rate, 5% warmup plus cosine decay, weight decay
`0.05`, per-epoch full validation, and checkpoints only at epochs
5/10/15/20/25.  Test remains locked and uncached.

The implementation adds `--no-proprio-target` to
`scripts/train_single_step_dynamics.py`.  After the subsequent no-input
extension, the targeted and full regression suites passed (11/11 and 43/43
tests).  The queued run directory and log are:

```text
outputs/single_step_dynamics/formal_native256_visual_only_100_per_task_stride1_b128_seed_0/
outputs/single_step_dynamics/formal_native256_visual_only_100_per_task_stride1_b128_seed_0/train.log
```

The detached queue is `vjepa2_native256_visual_only_queue`.  It waits for the
current `vjepa2_native256_train` tmux session to end and then validates that
the current summary contains exactly 25 epochs, 44,425 updates, no early stop,
the native-256 encoder, and the joint-target architecture.  It launches the
visual-only run only if all completion gates pass; an interrupted or failed
current run causes the queue to exit without consuming the GPUs.  The gate and
launch command are recorded in
`scripts/run_visual_only_after_native256.sh`.

To guard against a transient launch failure, this queue permits at most three
attempts separated by 120 seconds, but **only while no retained history,
summary, or epoch checkpoint exists**.  Once retained training progress exists,
it refuses to restart from scratch automatically.  There is no wall-clock
forced launch, so this run cannot intentionally overlap the current run.

## Queued completely proprio-free ablation

A third controlled run was implemented and queued after the conditioned
visual-only run.  Its model contract is strictly:

```text
inputs:  z_t, a_t, a_{t+1}, a_{t+2}, a_{t+3}
target:  z_{t+4}
```

It has no current-state token, no future-state head, and no proprio loss or
proprio metrics.  Training and evaluation pass `anchor_state=None`; the model
raises an error if a proprio tensor is supplied, preventing accidental state
leakage.  The common dataset still contains state arrays on CPU, but neither
current nor future proprioception is transferred to the model/GPU in this
run.

This run follows the conditioned visual-only run rather than the joint-target
run so the B-to-C comparison changes exactly one factor: whether `s_t` is an
input condition.  Shared seed-0 parameters are elementwise identical between
the two models.  Removing `state_input` and `state_type` reduces parameters
from 80,493,504 to 80,476,224 (17,280 parameters); all other architecture,
data, sampling, optimizer, schedule, validation, checkpoint, and locked-test
settings remain fixed.  The CLI contract requires both
`--no-proprio-target --no-proprio-input`; invalid combinations fail fast.

The implementation passed 11/11 targeted tests and 43/43 full regression
tests.  The queued run directory and log are:

```text
outputs/single_step_dynamics/formal_native256_no_proprio_100_per_task_stride1_b128_seed_0/
outputs/single_step_dynamics/formal_native256_no_proprio_100_per_task_stride1_b128_seed_0/train.log
```

The detached queue `vjepa2_native256_no_proprio_queue` waits for
`vjepa2_native256_visual_only_queue` to terminate.  A successful 25-epoch /
44,425-update visual-only summary with current-proprio conditioning is the
normal upstream condition.  If that run ultimately fails or is incomplete,
the third experiment is not lost: it records `upstream_status=failed` and
`fallback_launch=true` in its own `queue_context.json`, waits until all GPUs
are idle and no single-step trainer remains (up to 30 minutes), and then starts
independently.  Thus an upstream failure is explicit in the provenance but
does not block the scientifically useful no-proprio run.

The third run has the same at-most-three startup attempts, again only before
any history, summary, or epoch checkpoint has been retained.  No fixed start
time is used; both the dependency and GPU-idle checks prevent overlap.  The
gate, fallback provenance, idle check, and launch command are in
`scripts/run_no_proprio_after_visual_only.sh`.

Operational note (2026-08-04): the first automatic launch did not begin
training because the original `pgrep -f` idle check matched the long-lived tmux
server's historical command line even though both GPUs were empty.  It exited
without producing history or checkpoints.  The check was replaced with a
`/proc` scan that requires a real Python executable whose argument vector
contains `scripts/train_single_step_dynamics.py`, so tmux metadata cannot be a
false positive.  After syntax and empty-GPU checks, the completely proprio-free
run was relaunched at 11:21:39+02:00.  Its recorded upstream status is
`complete` with `fallback_launch=false`; both A40s entered active computation.

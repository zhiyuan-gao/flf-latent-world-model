# Frozen-encoder dynamics bake-off

This experiment compares frozen GR00T Eagle2 and frozen V-JEPA 2 under the
same four-task, four-horizon action-conditioned latent predictor.

## Fixed data contract

- Tasks: `PreSoakPan`, `KettleBoiling`, `LoadDishwasher`, `RinseSinkBasin`
- Split per task: 50 train / 10 validation / 20 test episodes
- Camera: `robot0_agentview_left`
- History: `t-12, t-8, t-4, t`
- Actions: `a[t:t+16]`, represented as four ordered 4-action blocks
- Targets: `t+4, t+8, t+12, t+16`
- No language, task id, Base action head, time warping, or future leakage

The generated manifest contains 28,296 train, 5,427 validation, and 10,657
test windows. Episode membership is shared exactly by both encoders.

## Commands

Prepare the shared manifest and normalized expert actions:

```bash
.venv-robocasa-gr00t/bin/python scripts/prepare_dynamics_bakeoff.py
```

Extract features on the two A40s. These commands may run concurrently:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-robocasa-gr00t/bin/python \
  scripts/extract_dynamics_features.py \
  --encoder gr00t --device cuda:0 --batch-size 16

CUDA_VISIBLE_DEVICES=1 .venv-vjepa2/bin/python \
  scripts/extract_dynamics_features.py \
  --encoder vjepa2 --device cuda:0 --batch-size 8
```

Each encoder is spatially pooled to 8x8, then independently projected to 256
dimensions using PCA and whitening fitted only on train episodes. Projected
features are float16 and resumably cached under
`outputs/dynamics_bakeoff/features/<encoder>/`.

Run one seed per GPU after both caches finish:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-robocasa-gr00t/bin/python \
  scripts/train_dynamics_bakeoff.py --encoder gr00t --device cuda:0 --seed 0

CUDA_VISIBLE_DEVICES=1 .venv-robocasa-gr00t/bin/python \
  scripts/train_dynamics_bakeoff.py --encoder vjepa2 --device cuda:0 --seed 0
```

Training supervises all four future times with teacher forcing and uses the
same two-step autoregressive training loss as the V-JEPA 2-AC recipe. Held-out
evaluation always performs the complete four-step open-loop rollout.

Repeat for seeds 1 and 2. The training environment does not load either
encoder; both predictors train solely from the common cache format.
The measured A40-safe default is batch 256 (about 23 GiB peak), a 19.23M
parameter predictor, 10 epochs, and a 5% learning-rate warmup.
The output is residual dynamics (`next = current + delta`) with a near-zero
delta-head initialization, so the common starting point is the persistence
baseline rather than an arbitrary absolute latent prediction.

Evaluate a best checkpoint:

```bash
.venv-robocasa-gr00t/bin/python scripts/evaluate_dynamics_bakeoff.py \
  --checkpoint outputs/dynamics_bakeoff/runs/gr00t/seed_0/best.pt \
  --split test --device cuda:0
```

## Go/no-go interpretation

The primary number is error normalized by the copy-current persistence
baseline; less than 1 means the model predicts the future better than simply
reusing the current latent. Correct actions must also beat zero and shuffled
actions. A model that beats persistence but is insensitive to actions is not
an action-consequence model and must not be used to guide a residual policy.

Raw losses must not be compared across GR00T and V-JEPA feature spaces.

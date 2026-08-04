# FLF Latent World Model for RoboCasa365

Research code for an action-conditioned latent world model in a hierarchical
RoboCasa365 video-guided policy. The current model predicts a frozen visual
representation four control steps into the future:

```text
(z_t, optional s_t, a_t, a_{t+1}, a_{t+2}, a_{t+3}) -> z_{t+4}
```

The repository contains data preparation, frozen-encoder feature extraction,
predictor training/evaluation, progress-tracking utilities, controlled
proprioception ablations, tests, and detailed experiment records. It does
**not** contain datasets, model weights, feature caches, third-party sources,
or trained checkpoints.

> Status: experimental. The reported predictors beat the copy-current
> persistence baseline on average, but have not yet passed the held-out
> future-closeness gate required for residual-policy or CEM use. See
> [`LATENT_WORLD_MODEL_EXPERIMENT_SUMMARY.md`](LATENT_WORLD_MODEL_EXPERIMENT_SUMMARY.md).

## Current benchmark

- Tasks: `PreSoakPan`, `KettleBoiling`, `LoadDishwasher`, `RinseSinkBasin`
- Data: RoboCasa365 target-human Composite-Seen demonstrations
- Camera: `robot0_agentview_left`
- Control rate: 20 Hz
- Model transition: four actions / 0.2 seconds
- Visual encoder: frozen V-JEPA2 ViT-g/16, native 256 checkpoint
- Visual grid: FP16 `16 x 16 x 1408`, with no pooling or PCA
- Predictor: width 960, depth 7, 12 heads, approximately 80.5M parameters
- Formal training: two GPUs, global batch 128, 25 epochs

Human300 is the 300-task **pretraining** dataset. It is useful for later
large-scale experiments, but it does not replace the four target-human task
snapshots used by the reported controlled experiments.

## Repository layout

```text
configs/                  portable fixed split metadata
scripts/                  preparation, extraction, training, and evaluation CLIs
src/dynamics/             datasets, encoders, predictors, losses, and metrics
src/progress/             video/progress localization modules
tests/                    CPU regression tests
*.md                      protocols, results, and resource manifests
```

Local/generated directories such as `data/`, `checkpoints/`, `outputs/`,
`third_party/`, and `.venv*` are intentionally ignored by Git.

## 1. Hardware and storage

The code is developed on Linux with Python 3.10, PyTorch 2.5.1 + CUDA 12.4,
and two NVIDIA A40 GPUs. Feature extraction and formal training can use two
GPUs, while unit tests and small audits can run on CPU.

Recommended free space:

| Configuration | Recommended free space |
| --- | ---: |
| Four-task data, weights, and V-JEPA2 cache | at least 300 GiB |
| V-JEPA2 and DINOv3 caches together | at least 400 GiB |

The measured native V-JEPA2 train/validation cache is 172.27 GiB. A planned
DINOv3 cache is estimated at about 125.29 GiB.

## 2. Python environment for latent dynamics

Create an isolated environment. Do not install into the system Python:

```bash
python3.10 -m venv .venv-dynamics
source .venv-dynamics/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements-dynamics.txt
```

Verify the installation:

```bash
python - <<'PY'
import cv2
import numpy
import pandas
import torch
import transformers

print("torch", torch.__version__, "CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available(), "GPUs", torch.cuda.device_count())
print("transformers", transformers.__version__)
PY

python -m pytest -q
```

`flash-attn`, RoboCasa, robosuite, and Isaac-GR00T are not required when
training a predictor from an existing feature cache.

## 3. Required datasets

Download these four official RoboCasa365 target-human snapshots:

| Task | Snapshot | Episodes | Frames |
| --- | --- | ---: | ---: |
| PreSoakPan | `20250809` | 501 | 395,501 |
| KettleBoiling | `20250814` | 501 | 228,349 |
| LoadDishwasher | `20250811` | 501 | 369,430 |
| RinseSinkBasin | `20250816` | 509 | 211,036 |

Expected layout:

```text
data/robocasa365/v1.0/target/composite/
├── PreSoakPan/20250809/lerobot/
├── KettleBoiling/20250814/lerobot/
├── LoadDishwasher/20250811/lerobot/
└── RinseSinkBasin/20250816/lerobot/
```

The four datasets total 2,012 Parquet episodes, 6,036 MP4 files, and
1,204,316 frames. Exact official download commands, Box identifiers, and a
read-only integrity checker are in
[`SECOND_SERVER_RESOURCE_SETUP.md`](SECOND_SERVER_RESOURCE_SETUP.md).

The optional Human300 dataset belongs under a separate pretraining path; do
not point the four-task target manifest at Human300.

## 4. Required and optional model weights

### V-JEPA2: required for the current experiment

- Hugging Face repo: `facebook/vjepa2-vitg-fpc64-256`
- Revision: `875c192b7b704b87d1e1d99345769632dd5f739a`
- `model.safetensors` SHA-256:
  `f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b`
- Expected model path: `checkpoints/vjepa2-vitg-fpc64-256/`

```bash
hf download facebook/vjepa2-vitg-fpc64-256 \
  README.md config.json model.safetensors video_preprocessor_config.json \
  --revision 875c192b7b704b87d1e1d99345769632dd5f739a \
  --local-dir checkpoints/vjepa2-vitg-fpc64-256
```

Do not use `vjepa2-vitg-fpc64-384` with a forced 256 crop. That historical
configuration produced a mislabeled cache and is not comparable to the
native-256 results.

### DINOv3: optional planned encoder ablation

- Hugging Face repo: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Revision: `ea8dc2863c51be0a264bab82070e3e8836b02d51`
- `model.safetensors` SHA-256:
  `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`
- Expected model path: `checkpoints/dinov3-vitl16-pretrain-lvd1689m/`

DINOv3 is gated on Hugging Face. Accept its license and authenticate before
downloading. The intended representation is the 256 patch tokens only,
reshaped to FP16 `16 x 16 x 1024`; exclude the CLS token and four register
tokens. The repository does not yet provide the finalized DINOv3 extractor or
dimension-metadata changes, so this checkpoint is not a drop-in replacement
for a V-JEPA2 cache.

### GR00T N1.5: optional Base VLA / later planning experiments

- Hugging Face repo: `robocasa/robocasa365_checkpoints`
- Subdirectory:
  `gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000`
- Revision: `14895998fe7c8f8f2441cc8957ec2c510302758b`
- Expected model path:
  `checkpoints/gr00t_n1-5_composite_seen_target_posttraining/checkpoint-60000/`

Only the two Safetensors shards, index, config, and experiment metadata are
needed. Exact file checksums are in `SECOND_SERVER_RESOURCE_SETUP.md`.

## 5. Optional simulator and Base-VLA source dependencies

These repositories are required only for RoboCasa simulation, Base GR00T
inference, and closed-loop policy experiments. Check out the pinned revisions:

| Component | Revision |
| --- | --- |
| `robocasa/robocasa` | `b4684e6ee37d377cc392e98302a6b916d588b415` |
| `ARISE-Initiative/robosuite` | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| `NVIDIA/Isaac-GR00T` | `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10` |
| `facebookresearch/vjepa2` | `204698b45b3712590f06245fbfba32d3be539812` |

Place them under `third_party/`. RoboCasa simulation additionally needs the
official kitchen asset packs and a generated `macros_private.py`. See
[`MVP_INSTALL_MANIFEST.md`](MVP_INSTALL_MANIFEST.md) for the verified runtime
boundary. The original combined runtime used Python 3.10.16, PyTorch
2.5.1+cu124, and flash-attn 2.7.4.post1.

## 6. Reproduce the fixed data split

The checked-in split file stores the exact 100 train / 10 validation / 20
locked-test episode IDs per task used by the native-256 experiments. Generate
portable manifests and small action/state episode caches on the new machine:

```bash
source .venv-dynamics/bin/activate

python scripts/prepare_checkvla_predictor_data.py \
  --data-root data/robocasa365/v1.0/target/composite \
  --output-dir outputs/single_step_dynamics/data_100_per_task_stride1 \
  --train-episodes 100 \
  --val-episodes 10 \
  --test-episodes 20 \
  --horizon 16 \
  --window-stride 1 \
  --reference-manifest configs/robocasa365_four_task_split_100.json
```

Expected window counts are 227,408 train, 22,139 validation, and 43,469 locked
test windows. Training and model selection must not read the test split.

## 7. Extract native V-JEPA2 features

Extract train and validation only. The following example assigns two tasks to
each GPU and can be run in two terminals:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_endpoint_predictor_features.py \
  --manifest-dir outputs/single_step_dynamics/data_100_per_task_stride1 \
  --checkpoint checkpoints/vjepa2-vitg-fpc64-256 \
  --output-root outputs/single_step_dynamics/data_100_per_task_stride1/features/vjepa2_native_256 \
  --tasks KettleBoiling LoadDishwasher \
  --splits train val --device cuda:0 --batch-size 8

CUDA_VISIBLE_DEVICES=1 python scripts/extract_endpoint_predictor_features.py \
  --manifest-dir outputs/single_step_dynamics/data_100_per_task_stride1 \
  --checkpoint checkpoints/vjepa2-vitg-fpc64-256 \
  --output-root outputs/single_step_dynamics/data_100_per_task_stride1/features/vjepa2_native_256 \
  --tasks PreSoakPan RinseSinkBasin \
  --splits train val --device cuda:0 --batch-size 8
```

The completed cache should contain 440 episode pairs and 256,587 frame rows.
Metadata must report `vjepa2-vitg-fpc64-256`, crop size 256, and observed
feature shape `[16,16,1408]`. Alternatively, transfer the verified 172.27 GiB
cache from an existing machine.

## 8. Train the current single-step predictor

Twenty-step two-GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/train_single_step_dynamics.py \
  --manifest-dir outputs/single_step_dynamics/data_100_per_task_stride1 \
  --feature-root outputs/single_step_dynamics/data_100_per_task_stride1/features/vjepa2_native_256 \
  --seed 0 --batch-size 128 \
  --learning-rate 3e-4 --weight-decay 0.05 --warmup-ratio 0.05 \
  --no-proprio-target --no-proprio-input \
  --max-epochs 1 --min-epochs 1 --disable-early-stopping \
  --max-steps-per-epoch 20 --checkpoint-epochs \
  --output-dir outputs/single_step_dynamics/smoke_no_proprio
```

Formal completely proprio-free run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/train_single_step_dynamics.py \
  --manifest-dir outputs/single_step_dynamics/data_100_per_task_stride1 \
  --feature-root outputs/single_step_dynamics/data_100_per_task_stride1/features/vjepa2_native_256 \
  --seed 0 --batch-size 128 \
  --learning-rate 3e-4 --weight-decay 0.05 --warmup-ratio 0.05 \
  --no-proprio-target --no-proprio-input \
  --max-epochs 25 --min-epochs 25 --disable-early-stopping \
  --checkpoint-epochs 5 10 15 20 25 \
  --output-dir outputs/single_step_dynamics/formal_native256_no_proprio_seed_0
```

Remove `--no-proprio-input` to retain current proprioception as a condition.
Remove both no-proprio flags and add `--proprio-weight 0.005` for joint visual
and future-proprio supervision.

## 9. Evaluation rules

Use complete held-out validation and report:

- normalized error relative to copy-current persistence;
- overall and visually dynamic `future_closer`;
- visual-delta cosine and RMS ratio;
- correct action versus zero and far same-task shuffled action;
- per-task metrics and the train/validation gap.

A normalized error below one is not sufficient. Do not use a checkpoint for
residual-policy training or CEM unless held-out predictions actually advance
toward the future while remaining action-sensitive.

## Documentation

- [`LATENT_WORLD_MODEL_EXPERIMENT_SUMMARY.md`](LATENT_WORLD_MODEL_EXPERIMENT_SUMMARY.md): chronological results and decisions
- [`SINGLE_STEP_DYNAMICS_EXPERIMENT.md`](SINGLE_STEP_DYNAMICS_EXPERIMENT.md): current one-step protocol
- [`ENDPOINT_PREDICTOR_EXPERIMENT.md`](ENDPOINT_PREDICTOR_EXPERIMENT.md): direct endpoint and causal multi-time experiments
- [`SECOND_SERVER_RESOURCE_SETUP.md`](SECOND_SERVER_RESOURCE_SETUP.md): exact downloads, checksums, and storage planning
- [`MVP_INSTALL_MANIFEST.md`](MVP_INSTALL_MANIFEST.md): simulator/Base-VLA installation record

## License

Project-owned code is released under the MIT License. Downloaded datasets,
weights, assets, and third-party repositories retain their own licenses and
are not redistributed here.

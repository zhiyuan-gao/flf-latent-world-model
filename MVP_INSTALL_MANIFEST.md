# MVP local installation manifest

Installation root: this repository root.

## Selected tasks and demonstrations

The official RoboCasa365 target-human LeRobot datasets for the four selected
Composite-Seen tasks are installed under `data/robocasa365/v1.0/target/composite`.

| Task | Dataset snapshot | Episodes | Frames | Cameras |
|---|---:|---:|---:|---:|
| PreSoakPan | 20250809 | 501 | 395,501 | 3 |
| KettleBoiling | 20250814 | 501 | 228,349 | 3 |
| LoadDishwasher | 20250811 | 501 | 369,430 | 3 |
| RinseSinkBasin | 20250816 | 509 | 211,036 | 3 |
| **Total** | | **2,012** | **1,204,316** | **6,036 MP4 files** |

Each dataset also includes the original state/XML extras and the per-frame
human subtask annotations used for oracle subtask-video selection.

RoboCasa's local `macros_private.py` points `DATASET_BASE_PATH` at this dataset
root, so the official dataset registry resolves these paths without moving or
duplicating the data.

## Checkpoints

- GR00T N1.5 official RoboCasa365 Composite-Seen target-posttraining
  `checkpoint-60000`: `checkpoints/gr00t_n1-5_composite_seen_target_posttraining/checkpoint-60000`
  (two safetensors shards, about 7.1 GiB).
- V-JEPA 2 ViT-G FPC64-384:
  `checkpoints/vjepa2-vitg-fpc64-384` (one safetensors file, about 3.9 GiB).

Only inference/fine-tuning model files and their configs were downloaded.
Optimizer, scheduler, RNG, trainer-state, eval-output, and duplicate original
PyTorch checkpoint files were intentionally omitted.

## Source and pinned revisions

| Component | Local path | Revision |
|---|---|---|
| RoboCasa | `third_party/robocasa` | `b4684e6ee37d377cc392e98302a6b916d588b415` |
| robosuite | `third_party/robosuite` | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| RoboCasa Isaac-GR00T fork | `third_party/Isaac-GR00T` | `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10` |
| V-JEPA 2 | `third_party/vjepa2` | `204698b45b3712590f06245fbfba32d3be539812` |

The six official RoboCasa asset packs are unpacked directly into
`third_party/robocasa/robocasa/models/assets`: textures, generative textures,
Lightwheel fixtures, Objaverse objects, AI-generated objects, and Lightwheel
objects.

## Runtime environment boundary

No packages were installed into the system Python. The installed GR00T source
is RoboCasa's official benchmark fork, which provides the PandaOmron data
configuration and `scripts/run_eval.py`. Start with one isolated benchmark
environment containing RoboCasa and this fork; split V-JEPA2 into another
environment only if its dependencies conflict. The official evaluator uses an
inference server/client internally but can launch both from one command.

The current MVP uses ground-truth subtask clips, so no video-generation model
is part of this installation.

## Verified Base-only runtime

The combined RoboCasa + GR00T inference environment is installed at
`.venv-robocasa-gr00t` with Python 3.10.16, PyTorch 2.5.1+cu124 and the official
prebuilt `flash-attn` 2.7.4.post1 wheel. GR00T's declared runtime pins take
precedence over the older transitive LeRobot / RoboCasa package metadata in
this environment. The actual PandaOmron reset, EGL render, checkpoint load,
model inference, action execution and H.264 recording paths have all been
verified end to end on an NVIDIA A40.

The project-owned evaluator `scripts/run_base_only_four_tasks.py` wraps the
official policy server and simulation client without changing the benchmark
fork. It enforces the four-task whitelist and records the exact run config,
one MP4 and one `stats.json` per task. The official evaluation default executes
all 16 predicted actions before observing and replanning.

```bash
cd /path/to/flf-latent-world-model
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0
export NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_DISABLE_TELEMETRY=1
.venv-robocasa-gr00t/bin/python scripts/run_base_only_four_tasks.py \
  --n-episodes 1 \
  --n-envs 1 \
  --n-action-steps 16 \
  --output-dir outputs/base_only_official_chunk16
```

### 2026-08-01 engineering smoke result

| Task | Episode result | Official horizon | Wall time | Decoded video frames |
|---|---:|---:|---:|---:|
| PreSoakPan | success | 2,400 | 233.45 s | 1,200 |
| KettleBoiling | success | 1,500 | 122.86 s | 750 |
| LoadDishwasher | failure | 1,800 | 119.47 s | 900 |
| RinseSinkBasin | success | 1,350 | 110.48 s | 675 |

All four episodes completed without infrastructure errors and all four MP4
files decode successfully. This 3/4 outcome is only an engineering smoke test,
not a success-rate estimate: there is only one episode per task and the stock
official runner does not expose a fixed reset seed. Statistical baseline runs
must add explicit reset-seed control and use the planned 20--50 episodes per
task. The verified artifacts are under
`outputs/base_only_official_chunk16/evals/target`.

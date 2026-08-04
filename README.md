# RoboCasa365 潜空间世界模型

本仓库用于训练 action-conditioned latent world model，并接入 RoboCasa365
子任务视频引导的闭环策略。当前 predictor 学习以下四步状态转移：

```text
(z_t, [可选] s_t, a_t, a_{t+1}, a_{t+2}, a_{t+3}) -> z_{t+4}
```

## 1. 克隆仓库

```bash
git clone git@github.com:zhiyuan-gao/flf-latent-world-model.git
cd flf-latent-world-model
```

## 2. 当前实验配置

| 项目 | 设置 |
| --- | --- |
| RoboCasa365 任务 | `PreSoakPan`、`KettleBoiling`、`LoadDishwasher`、`RinseSinkBasin` |
| 相机视角 | `robot0_agentview_left` |
| 控制频率 | 20 Hz |
| 预测目标 | `t -> t+4`，对应 0.2 秒 |
| 动作 token | 4 个有顺序的 12 维 token |
| V-JEPA2 特征 | FP16 `[16,16,1408]` |
| 预测器 | 宽度 960、深度 7、12 个注意力头，约 80.5M 参数 |
| 训练 | 2 张 GPU、全局 batch 128、25 个 epoch |

## 3. 硬件与磁盘

已验证环境：

- Linux；
- Python 3.10；
- PyTorch 2.5.1 + CUDA 12.4；
- 2×NVIDIA A40；
- V-JEPA2 正式 cache 约 172.27 GiB；
- V-JEPA2 全流程建议准备至少 300 GiB 可用空间；
- 同时保存 V-JEPA2 与 DINOv3 cache 建议准备至少 400 GiB。

开始前检查：

```bash
python3 --version
nvidia-smi
df -h .
```

## 4. 数据

### 4.1 四任务 target-human 数据

下载以下 RoboCasa365 target-human snapshots：

| 任务 | 数据版本 | 轨迹数 | 帧数 |
| --- | --- | ---: | ---: |
| `PreSoakPan` | `20250809` | 501 | 395,501 |
| `KettleBoiling` | `20250814` | 501 | 228,349 |
| `LoadDishwasher` | `20250811` | 501 | 369,430 |
| `RinseSinkBasin` | `20250816` | 509 | 211,036 |

目录结构：

```text
data/robocasa365/v1.0/target/composite/
├── PreSoakPan/20250809/lerobot/
├── KettleBoiling/20250814/lerobot/
├── LoadDishwasher/20250811/lerobot/
└── RinseSinkBasin/20250816/lerobot/
```

合计应有 2,012 个 Parquet episodes、6,036 个 MP4 和 1,204,316 帧。官方 Box
下载命令和数据完整性检查见
[`SECOND_SERVER_RESOURCE_SETUP.md`](SECOND_SERVER_RESOURCE_SETUP.md)。

### 4.2 Human300

Human300 用于后续扩大任务和场景覆盖的预训练实验。按照 RoboCasa365 官方
`pretrain_human300` 配置保存数据，并将其与四任务 target-human 数据分别管理。

- 官方文档：<https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html>
- 数据来源：RoboCasa365 Human300 pretraining dataset

## 5. 模型权重

先安装 Hugging Face 下载工具：

```bash
python3 -m venv .venv-download
source .venv-download/bin/activate
python -m pip install --upgrade pip huggingface_hub
hf auth login
```

### 5.1 V-JEPA2 ViT-g/16 native 256

当前正式 latent world model 使用：

- 模型仓库：`facebook/vjepa2-vitg-fpc64-256`
- 固定版本：`875c192b7b704b87d1e1d99345769632dd5f739a`
- `model.safetensors` SHA-256：
  `f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b`

```bash
mkdir -p checkpoints/vjepa2-vitg-fpc64-256

hf download facebook/vjepa2-vitg-fpc64-256 \
  README.md config.json model.safetensors video_preprocessor_config.json \
  --revision 875c192b7b704b87d1e1d99345769632dd5f739a \
  --local-dir checkpoints/vjepa2-vitg-fpc64-256

sha256sum checkpoints/vjepa2-vitg-fpc64-256/model.safetensors
```

预处理固定为短边 resize 到 292，再中心裁剪到 256×256，输出
`16×16×1408` patch-token grid。

### 5.2 DINOv3 ViT-L/16

DINOv3 用于相同数据、split、视角和 predictor 下的 encoder 对照实验：

- 模型仓库：`facebook/dinov3-vitl16-pretrain-lvd1689m`
- 固定版本：`ea8dc2863c51be0a264bab82070e3e8836b02d51`
- `model.safetensors` SHA-256：
  `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`

```bash
mkdir -p checkpoints/dinov3-vitl16-pretrain-lvd1689m

hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  README.md LICENSE.md config.json model.safetensors preprocessor_config.json \
  --revision ea8dc2863c51be0a264bab82070e3e8836b02d51 \
  --local-dir checkpoints/dinov3-vitl16-pretrain-lvd1689m

sha256sum checkpoints/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
```

DINOv3 表征协议为 256×256 输入、256 个 patch tokens、FP16
`16×16×1024`；CLS 与 4 个 register tokens 单独处理。

### 5.3 GR00T N1.5 Composite-Seen checkpoint

GR00T 用作完整研究框架中的 Base VLA 和候选 action 生成器：

- 模型仓库：`robocasa/robocasa365_checkpoints`
- 固定版本：`14895998fe7c8f8f2441cc8957ec2c510302758b`
- 权重子目录：
  `gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000`

```bash
export GR00T_REPO_DIR=checkpoints/robocasa365_checkpoints
export GR00T_PREFIX=gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000

mkdir -p "$GR00T_REPO_DIR"

hf download robocasa/robocasa365_checkpoints \
  "$GR00T_PREFIX/config.json" \
  "$GR00T_PREFIX/experiment_cfg/metadata.json" \
  "$GR00T_PREFIX/model.safetensors.index.json" \
  "$GR00T_PREFIX/model-00001-of-00002.safetensors" \
  "$GR00T_PREFIX/model-00002-of-00002.safetensors" \
  --revision 14895998fe7c8f8f2441cc8957ec2c510302758b \
  --local-dir "$GR00T_REPO_DIR"

mkdir -p checkpoints/gr00t_n1-5_composite_seen_target_posttraining
ln -s "$(pwd)/$GR00T_REPO_DIR/$GR00T_PREFIX" \
  checkpoints/gr00t_n1-5_composite_seen_target_posttraining/checkpoint-60000
```

五个文件的 SHA-256 见
[`SECOND_SERVER_RESOURCE_SETUP.md`](SECOND_SERVER_RESOURCE_SETUP.md)。

## 6. 潜空间动力学 Python 环境

```bash
python3.10 -m venv .venv-dynamics
source .venv-dynamics/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements-dynamics.txt
```

验证环境：

```bash
python - <<'PY'
import cv2
import numpy
import pandas
import torch
import transformers

print("torch", torch.__version__)
print("CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("GPU count", torch.cuda.device_count())
print("transformers", transformers.__version__)
PY

python -m pytest -q
```

## 7. RoboCasa + GR00T 环境

完整模拟和 Base VLA 使用以下源码 revision：

| 组件 | 固定版本 |
| --- | --- |
| `robocasa/robocasa` | `b4684e6ee37d377cc392e98302a6b916d588b415` |
| `ARISE-Initiative/robosuite` | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| `NVIDIA/Isaac-GR00T` | `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10` |
| `facebookresearch/vjepa2` | `204698b45b3712590f06245fbfba32d3be539812` |

```bash
mkdir -p third_party

git clone https://github.com/ARISE-Initiative/robosuite.git third_party/robosuite
git -C third_party/robosuite checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c

git clone https://github.com/robocasa/robocasa.git third_party/robocasa
git -C third_party/robocasa checkout b4684e6ee37d377cc392e98302a6b916d588b415

git clone https://github.com/NVIDIA/Isaac-GR00T.git third_party/Isaac-GR00T
git -C third_party/Isaac-GR00T checkout 9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10

git clone https://github.com/facebookresearch/vjepa2.git third_party/vjepa2
git -C third_party/vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
```

建立环境并安装：

```bash
python3.10 -m venv .venv-robocasa-gr00t
source .venv-robocasa-gr00t/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e third_party/robosuite
python -m pip install -e third_party/robocasa
python -m pip install -e third_party/Isaac-GR00T
```

配置 RoboCasa 并下载 kitchen assets：

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

已验证的组合环境使用 Python 3.10.16、PyTorch 2.5.1+cu124 和
flash-attn 2.7.4.post1。详细安装记录见
[`MVP_INSTALL_MANIFEST.md`](MVP_INSTALL_MANIFEST.md)。

## 8. 生成固定数据划分

仓库中的
`configs/robocasa365_four_task_split_100.json` 保存了正式实验使用的每任务
100 train / 10 validation / 20 test episode IDs。

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

预期生成：

- 227,408 train windows；
- 22,139 validation windows；
- 43,469 test windows。

训练与模型选择使用 train/validation；test 在实验方案冻结后评估。

## 9. 提取 V-JEPA2 特征缓存

两张 GPU 分别处理两个任务：

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

正式 cache 应包含：

- 440 个 train/validation episode feature pairs；
- 256,587 个 frame rows；
- FP16 `[N,16,16,1408]`；
- metadata 中记录 native-256 checkpoint、crop size 256 和 feature shape。

## 10. 训练预测器

先运行 20-step 双卡 smoke：

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

正式 no-proprio run：

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

带 current proprio condition 的视觉预测 run 使用
`--no-proprio-target`。联合预测 visual 与 future proprio 的 run 使用
`--proprio-weight 0.005`。

## 11. 许可证

项目代码采用 MIT 许可证。

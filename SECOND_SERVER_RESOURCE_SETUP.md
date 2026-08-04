# 第二台服务器：数据与模型预下载说明

更新时间：2026-08-04

本文只负责在另一台服务器上准备当前研究所需的**官方数据和预训练模型**。
资源校验完成后，再单独同步项目代码、固定的数据 split 和 feature cache。

## 1. 当前需要准备什么

### 1.1 当前 latent world model 必需

| 资源 | 固定版本 | 用途 | 本机实测大小 |
| --- | --- | --- | ---: |
| RoboCasa365 target-human `PreSoakPan` | `20250809` | 轨迹、动作、proprio、视频 | 1.218 GiB |
| RoboCasa365 target-human `KettleBoiling` | `20250814` | 同上 | 0.774 GiB |
| RoboCasa365 target-human `LoadDishwasher` | `20250811` | 同上 | 1.542 GiB |
| RoboCasa365 target-human `RinseSinkBasin` | `20250816` | 同上 | 0.706 GiB |
| V-JEPA2 ViT-g/16（原生 256） | `facebook/vjepa2-vitg-fpc64-256` | 冻结视觉 encoder | 3.854 GiB |

四个 RoboCasa365 数据集共 2,012 条 episode、1,204,316 帧、6,036 个 MP4，
解压后合计约 4.241 GiB。

> **重要：**当前实验固定使用原生 `vjepa2-vitg-fpc64-256` checkpoint。官方预处理
> 先把短边缩放到 292，再中心裁剪到 256×256，输出 `16×16×1408` token grid。
> 不得使用 `vjepa2-vitg-fpc64-384` 权重配合 256 crop；旧实验曾出现过这种配置继承
> 错误，其 feature cache 已作废。

### 1.2 整体研究框架需要

| 资源 | 固定版本 | 用途 | 本机实测大小 |
| --- | --- | --- | ---: |
| GR00T N1.5 Composite-Seen target post-training | `checkpoint-60000` | Base VLA、候选 action、CEM/residual 实验 | 7.065 GiB |

需要的文件是两个 safetensors shard、config、index 和实验统计 metadata。

### 1.3 后续 DINOv3 encoder 对照实验

| 资源 | 固定版本 | 用途 | 官方文件大小 |
| --- | --- | --- | ---: |
| DINOv3 ViT-L/16（LVD-1689M） | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 冻结视觉 encoder 对照 | 1.129 GiB |

DINOv3 不是当前 V-JEPA2 正式训练的依赖；现在预下载它，是为了后续在相同四任务、
相同 split、相同单视角、相同 `t→t+4` 目标和相同 predictor 下进行受控 encoder
对照。选择 ViT-L/16、256 输入，是因为 JEPA-WMs 在 DROID/RoboCasa 配置中采用的
就是这一组合；参考其
[`DROID & RoboCasa` 配置表](https://github.com/facebookresearch/jepa-wms#pretrained-models)。

计划中的 DINOv3 表征为 float16 `16×16×1024` **patch tokens**。不得把 CLS token
或 4 个 register tokens 写入空间网格，也暂不做 pooling、PCA 或多层特征拼接。
当前训练代码仍固定接收 V-JEPA2 的 `16×16×1408`；DINOv3 只能在后续完成
`feature_dim` 元数据化和专用提取器后使用，不能直接套用现有 V-JEPA2 cache。

## 2. 服务器和磁盘检查

建议把所有预下载资源放在一个独立目录。以下路径必须按新服务器实际情况修改，
不要直接使用系统根目录或用户 home 目录作为递归操作目标。

```bash
export FLF_RESOURCE_ROOT=/path/to/large_disk/flf_resources
mkdir -p "$FLF_RESOURCE_ROOT/data"
mkdir -p "$FLF_RESOURCE_ROOT/models"
mkdir -p "$FLF_RESOURCE_ROOT/download_tools"

df -h "$FLF_RESOURCE_ROOT"
free -h
nvidia-smi
```

空间规划：

| 阶段 | 预计占用 |
| --- | ---: |
| 仅四任务数据 + V-JEPA2 | 约 8.1 GiB |
| 再加入 DINOv3 ViT-L/16 权重 | 约 9.3 GiB（累计） |
| 再加入 GR00T | 约 16.3 GiB（累计） |
| 当前 V-JEPA2 train+val feature cache | 再增加 172.27 GiB |
| 未来 DINOv3 train+val feature cache | 再增加约 125.29 GiB |
| 每个 encoder 的五个 predictor checkpoint | 约 1.5 GiB |

如果要在新服务器完整复跑当前实验，建议开始前至少准备 **220 GiB 可用空间**；
考虑临时文件、代码环境和后续实验，建议留 **300 GiB 以上**。
如果要同时保留 V-JEPA2 与 DINOv3 两套完整 cache 和训练 checkpoint，实际内容约
317 GiB，建议至少准备 **350 GiB**，较稳妥的是 **400 GiB 以上**。

当前正式配置使用两张 GPU；在 2×A40 上 global batch 128（每卡 64）的实测峰值
allocated CUDA memory 为 5.84 GiB/卡。新机器最终仍需做一次本地 smoke，不能只按
显存数字假定吞吐和 DataLoader 一定一致。

## 3. 安装轻量下载工具

这里只建立独立的下载环境，不安装本项目依赖：

```bash
python3 -m venv "$FLF_RESOURCE_ROOT/download_tools/venv"
source "$FLF_RESOURCE_ROOT/download_tools/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install --upgrade huggingface_hub

export HF_HOME="$FLF_RESOURCE_ROOT/download_tools/hf_cache"
hf --help >/dev/null
```

V-JEPA2 和 GR00T 资源目前可公开下载。DINOv3 官方 Hugging Face 仓库是
`manual gated`：必须先用浏览器登录，打开
[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)，
同意 DINOv3 license 并提交访问申请，然后在服务器登录同一账号：

```bash
hf auth login
hf auth whoami
```

未完成网页授权时，即使 token 有效也会收到 HTTP 401/403；不要改用来源不明的镜像
绕过 license。非交互式服务器可通过受权限保护的 `HF_TOKEN` 登录，但不要把 token
写进本文档、shell history 或项目仓库。

## 4. 下载四个 RoboCasa365 数据集

数据使用 RoboCasa 官方 Box 链接。下面四个 ID 来自 RoboCasa revision
`b4684e6ee37d377cc392e98302a6b916d588b415` 中的官方
[`box_links_ds.json`](https://github.com/robocasa/robocasa/blob/b4684e6ee37d377cc392e98302a6b916d588b415/robocasa/models/assets/box_links/box_links_ds.json)。

目标布局固定为：

```text
$FLF_RESOURCE_ROOT/data/robocasa365/v1.0/target/composite/
├── PreSoakPan/20250809/lerobot/
├── KettleBoiling/20250814/lerobot/
├── LoadDishwasher/20250811/lerobot/
└── RinseSinkBasin/20250816/lerobot/
```

执行：

```bash
set -euo pipefail

export ROBOCASA_DATA_ROOT="$FLF_RESOURCE_ROOT/data/robocasa365/v1.0/target/composite"
mkdir -p "$ROBOCASA_DATA_ROOT"

download_robocasa_task() {
  local task_name="$1"
  local snapshot="$2"
  local box_id="$3"
  local snapshot_dir="$ROBOCASA_DATA_ROOT/$task_name/$snapshot"
  local archive="$snapshot_dir/lerobot.tar"

  mkdir -p "$snapshot_dir"
  if test -f "$snapshot_dir/lerobot/meta/info.json"; then
    echo "Already present: $task_name $snapshot"
    return 0
  fi

  # Box 对这几份文件的 Range/HEAD 支持不一致，因此失败后重新下载 .part，
  # 不使用 curl --continue-at。四个完整 GET 直链已于 2026-08-04 验证为 HTTP 200。
  curl --fail --location \
    --retry 5 --retry-all-errors \
    "https://utexas.box.com/shared/static/${box_id}.tar" \
    --output "${archive}.part"
  mv "${archive}.part" "$archive"
  tar --extract --file "$archive" --directory "$snapshot_dir"
  test -f "$snapshot_dir/lerobot/meta/info.json"

  # 官方下载器解压成功后也会删除 tar；如需保留原始压缩包，可注释下一行。
  rm "$archive"
}

download_robocasa_task PreSoakPan 20250809 \
  krqwe33yytnrse06xchr5e41sap41xfv
download_robocasa_task KettleBoiling 20250814 \
  r3rwnzdw6caab3vivwv6uknl9sr8ru6j
download_robocasa_task LoadDishwasher 20250811 \
  k1qxg8rgjs0le1cnv98xylysh9t0dd8z
download_robocasa_task RinseSinkBasin 20250816 \
  blk33oaca11933xgre1sxemmtluo46eo
```

官方替代方法是在同步 RoboCasa 代码后运行：

```bash
python -m robocasa.scripts.download_datasets \
  --tasks PreSoakPan KettleBoiling LoadDishwasher RinseSinkBasin \
  --split target --source human
```

但本阶段还不整理代码，所以优先使用上面的四个直接下载命令。

## 5. 校验 RoboCasa365 数据

执行以下只读检查：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROBOCASA_DATA_ROOT"])
expected = {
    "PreSoakPan": ("20250809", 501, 395501),
    "KettleBoiling": ("20250814", 501, 228349),
    "LoadDishwasher": ("20250811", 501, 369430),
    "RinseSinkBasin": ("20250816", 509, 211036),
}

for task, (snapshot, episodes, frames) in expected.items():
    dataset = root / task / snapshot / "lerobot"
    info = json.loads((dataset / "meta/info.json").read_text())
    parquet_count = len(list((dataset / "data").glob("*/episode_*.parquet")))
    video_count = len(list((dataset / "videos").glob("*/*/episode_*.mp4")))
    assert info["total_episodes"] == episodes, (task, info["total_episodes"])
    assert info["total_frames"] == frames, (task, info["total_frames"])
    assert info["fps"] == 20, (task, info["fps"])
    assert parquet_count == episodes, (task, parquet_count)
    assert video_count == episodes * 3, (task, video_count)
    print(task, "OK", episodes, "episodes", frames, "frames", video_count, "videos")
PY

du -sh "$ROBOCASA_DATA_ROOT"/*
```

期望总计：2,012 个 parquet episode、6,036 个 MP4、1,204,316 帧。

## 6. 下载并校验 V-JEPA2

官方模型页：
[`facebook/vjepa2-vitg-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256)

固定 Hugging Face revision：
`875c192b7b704b87d1e1d99345769632dd5f739a`

```bash
export VJEPA_DIR="$FLF_RESOURCE_ROOT/models/vjepa2-vitg-fpc64-256"
mkdir -p "$VJEPA_DIR"

hf download facebook/vjepa2-vitg-fpc64-256 \
  README.md config.json model.safetensors video_preprocessor_config.json \
  --revision 875c192b7b704b87d1e1d99345769632dd5f739a \
  --local-dir "$VJEPA_DIR"
```

校验关键文件：

```bash
cd "$VJEPA_DIR"
sha256sum -c <<'EOF'
f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b  model.safetensors
d76799e52c1a5cf6b1e9204857232dd8047b034a9538e89e0c215df6859210c2  config.json
d2fab4418fc0390b62c4cd72ade56908a7929f80c62288adbe10dd8d23421227  video_preprocessor_config.json
EOF
du -sh "$VJEPA_DIR"
```

预期 `model.safetensors` 大小为 4,138,311,608 bytes。

还应确认 `config.json` 同时包含 `"crop_size": 256` 和 `"image_size": 256`，
`video_preprocessor_config.json` 包含短边 292、中心裁剪 256。代码会拒绝 checkpoint
分辨率与请求 crop 不一致的组合，防止再次误用 384 权重。

## 7. 下载并校验 DINOv3 ViT-L/16

官方模型页：
[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

固定 Hugging Face revision：
`ea8dc2863c51be0a264bab82070e3e8836b02d51`

先完成第 3 节的 gated-access 授权，再执行：

```bash
export DINOV3_DIR="$FLF_RESOURCE_ROOT/models/dinov3-vitl16-pretrain-lvd1689m"
mkdir -p "$DINOV3_DIR"

hf auth whoami
hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  README.md LICENSE.md config.json model.safetensors preprocessor_config.json \
  --revision ea8dc2863c51be0a264bab82070e3e8836b02d51 \
  --local-dir "$DINOV3_DIR"
```

校验模型权重和关键结构：

```bash
cd "$DINOV3_DIR"
sha256sum -c <<'EOF'
dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  model.safetensors
EOF

test "$(stat -c %s model.safetensors)" = 1212559808

python - <<'PY'
import json

cfg = json.load(open("config.json"))
assert cfg["model_type"] == "dinov3_vit", cfg
assert cfg["hidden_size"] == 1024, cfg
assert cfg["patch_size"] == 16, cfg
assert cfg["num_register_tokens"] == 4, cfg
print("DINOv3 config OK")
PY

du -sh "$DINOV3_DIR"
```

需要特别区分“checkpoint 默认配置”和“本实验输入协议”：官方 config 的
`image_size` 是 224，但 ViT 支持 256 输入；本实验按照 DINOv3 官方 LVD transform
和 JEPA-WMs 的 RoboCasa/DROID 设置，显式 resize 到 `256×256`，使用 ImageNet
mean/std 归一化，然后只提取 256 个 patch tokens，reshape 为 `16×16×1024`。
不能直接采用会输出 224 crop 的默认 processor，也不能把 CLS/register tokens
拼进网格。官方 transform 参考
[`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3#image-transforms)。

## 8. 建议预取 GR00T N1.5 checkpoint

官方目录：
[`target_posttraining/composite_seen/checkpoint-60000`](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000)

固定仓库 revision：
`14895998fe7c8f8f2441cc8957ec2c510302758b`

```bash
export GR00T_REPO_DIR="$FLF_RESOURCE_ROOT/models/robocasa365_checkpoints"
export GR00T_PREFIX="gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
mkdir -p "$GR00T_REPO_DIR"

hf download robocasa/robocasa365_checkpoints \
  "$GR00T_PREFIX/config.json" \
  "$GR00T_PREFIX/experiment_cfg/metadata.json" \
  "$GR00T_PREFIX/model.safetensors.index.json" \
  "$GR00T_PREFIX/model-00001-of-00002.safetensors" \
  "$GR00T_PREFIX/model-00002-of-00002.safetensors" \
  --revision 14895998fe7c8f8f2441cc8957ec2c510302758b \
  --local-dir "$GR00T_REPO_DIR"

export GR00T_CHECKPOINT="$GR00T_REPO_DIR/$GR00T_PREFIX"
```

校验：

```bash
cd "$GR00T_CHECKPOINT"
sha256sum -c <<'EOF'
6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526  config.json
0c4a350867f621ed3192f81c282eb7d23c05f275f347fd0deed3531afa29dbcf  experiment_cfg/metadata.json
bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746  model.safetensors.index.json
672b8e49d32ff124e13c3c4e4e70380ab29cd25f013ff41db768639347f8057e  model-00001-of-00002.safetensors
95c52f05a00141ce4e433305a8ccfa41fd219e8d988f2a4c991f3f4f5cdb677c  model-00002-of-00002.safetensors
EOF
du -sh "$GR00T_CHECKPOINT"
```

预期两个 shard 大小分别为 4,999,367,032 和 2,586,705,312 bytes。

## 9. Feature cache 如何处理

### 9.1 当前 V-JEPA2 cache

当前实验将重新生成的 cache 目标是：

```text
outputs/single_step_dynamics/data_100_per_task_stride1/
```

其中 train+validation V-JEPA2 feature cache 应满足：

- 100 train episodes/task；
- 原来的 10 validation episodes/task；
- stride 1；
- float16 `[N,16,16,1408]`；
- 440 个 train+validation episode feature pair；
- 本机实测 172.27 GiB；
- test feature 没有缓存。

同步前必须检查 cache 根目录下的 `metadata_*.json`：`checkpoint` 必须指向
`vjepa2-vitg-fpc64-256`，`crop_size` 必须为 256，`observed_feature_shape` 必须为
`[16,16,1408]`。任何记录 `vjepa2-vitg-fpc64-384` 的 cache 都不能使用。

代码和固定 split 同步时有两种准备方式：

1. 原生 256 cache 已完成并通过 metadata 校验，可从当前服务器直接 `rsync`，保证
   与当前实验逐文件一致；
2. 同步固定 manifest/split 后，在新服务器用同一个 V-JEPA2 checkpoint 重新提取。

不能只根据“100/task”重新随机划分，因为当前 100/task split 是旧 50/task split 的
嵌套扩展，validation/test episode 被固定保留。后续必须连同 manifest 一起迁移或使用
同样的 reference manifest 重建。

### 9.2 未来 DINOv3 cache

DINOv3 cache 尚未生成，也不能由当前 V-JEPA2 cache 转换得到。后续完成 DINOv3
提取器后，必须复用同一个 100/task stride-1 manifest，预期满足：

- 同样的 400 train、40 validation episode；
- 同样的 256,587 个已缓存 frame row；
- 输入严格为 256×256；
- 去除 CLS 和 4 个 register tokens；
- float16 `[N,16,16,1024]`；
- 不缓存 locked test；
- 预计约 125.29 GiB，按当前 V-JEPA2 cache 的实测体积和通道比
  `172.269 × 1024 / 1408` 计算。

这 125.29 GiB 是尺寸推算，不是当前服务器的已生成实测值。正式 cache 必须在
metadata 中记录 DINOv3 repo、固定 revision、权重 SHA-256、输入 transform、
register-token 处理和 observed feature shape，并再次运行全量 loader audit。

## 10. 下载完成后的最终布局

```text
$FLF_RESOURCE_ROOT/
├── data/
│   └── robocasa365/v1.0/target/composite/
│       ├── PreSoakPan/20250809/lerobot/
│       ├── KettleBoiling/20250814/lerobot/
│       ├── LoadDishwasher/20250811/lerobot/
│       └── RinseSinkBasin/20250816/lerobot/
├── models/
│   ├── vjepa2-vitg-fpc64-256/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── video_preprocessor_config.json
│   ├── dinov3-vitl16-pretrain-lvd1689m/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── preprocessor_config.json
│   └── robocasa365_checkpoints/
│       └── gr00t_n1-5/foundation_model_learning/
│           └── target_posttraining/composite_seen/checkpoint-60000/
└── download_tools/
    ├── venv/
    └── hf_cache/
```

## 11. 配置完成后需要记录

在开始同步项目代码前，保存以下输出：

```bash
date
hostname
nvidia-smi
df -h "$FLF_RESOURCE_ROOT"
du -sh "$ROBOCASA_DATA_ROOT"
du -sh "$VJEPA_DIR"
du -sh "$DINOV3_DIR"
du -sh "$GR00T_CHECKPOINT"
```

并确认：

- 四个数据集的第 5 节检查全部输出 `OK`；
- V-JEPA2 三个关键文件 checksum 全部 `OK`；
- DINOv3 gated access 可用，模型 checksum 和 config 检查全部 `OK`；
- GR00T 五个关键文件 checksum 全部 `OK`；
- 只复跑当前 V-JEPA2 实验时至少还剩 220 GiB，最好 300 GiB；
- 同时保留两套 encoder cache 时至少准备 350 GiB，最好 400 GiB 以上。

满足这些条件后，再进行项目代码、Python/CUDA 环境、固定 split 和 feature cache 的迁移。

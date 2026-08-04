# Latent world model 实验总结

更新时间：2026-08-04 12:39（Europe/Berlin）
项目目录：本仓库根目录

## 1. 文档范围

本文汇总当前研究中已经实际运行的 latent world model 实验，包括：

- 直接预测 `t+16` 的 endpoint predictor；
- 因果多时间目标和 V-JEPA2-AC normalization/loss 消融；
- 预测 `t+4` 的 single-step dynamics predictor；
- future proprio target、current proprio condition 的逐项消融；
- 与上述实验直接相关的 reproduction、gradient audit 和 smoke test。

本文不汇总 progress tracker、Base VLA 或 goal-video model 的实验结果。详细协议仍以
`ENDPOINT_PREDICTOR_EXPERIMENT.md` 和 `SINGLE_STEP_DYNAMICS_EXPERIMENT.md`
为准；本文用于快速比较实验结果和决定下一步。

## 2. 统一理解指标

核心视觉指标如下：

- `visual_normalized_error = MSE(pred, z_future) / MSE(z_t, z_future)`。
  persistence（直接复制当前状态）为 `1`，ground-truth future 为 `0`。
  小于 `1` 只说明平均误差优于复制当前状态，不代表预测已经真正到达未来。
- `future_closer` 是预测比 `z_t` 更接近 `z_future` 的样本比例。它要求预测跨过
  当前状态和未来状态之间的中点，是当前最重要的时间推进 gate。
- `correct beats zero/shuffle` 检查模型是否真正使用 action。动作敏感不等于动力学
  已学成，但动作不敏感的模型一定不能用于 CEM 或 residual action ranking。
- `dynamic future_closer` 只在视觉变化较大的固定子集上计算，用于排除大量近静态
  窗口掩盖失败。

当前 acceptance gate 是：固定 train diagnostic 和 held-out validation 都应明显朝
目标未来推进，同时保持正确动作优于 zero/shuffled action。在此之前不应把 predictor
冻结给 residual policy 或 CEM。

## 3. 一个重要的 encoder 追溯修正

早期 endpoint 系列以及最初的 single-step 实验使用的旧 cache 目录虽然包含
`native_256` 字样，但 metadata 后来证实其实际 checkpoint 是
`facebook/vjepa2-vitg-fpc64-384`，只是把输入强制 crop 到 256。这些结果仍可用于诊断
优化、监督形式和 temporal contraction，但不能称为原生 V-JEPA2-256 结果。

2026-08-04 重新提取的正式 cache 才使用：

- checkpoint：`facebook/vjepa2-vitg-fpc64-256`；
- revision：`875c192b7b704b87d1e1d99345769632dd5f739a`；
- native input/crop：256；
- latent grid：FP16 `[16, 16, 1408]`，不做空间池化或 PCA；
- 440 个 train/validation episode pairs，256,587 个 frame rows；
- cache 大小 172.27 GiB；
- 227,408 train 和 22,139 validation windows，缺失 feature 数为 0。

因此，本文将旧 encoder 结果标为“历史诊断”，将最后三个 native-256 runs 作为当前
主比较。

## 4. 核心实验总表

| ID | 实验 | 数据和目标 | 状态 | 最佳 val norm | 时间推进结果 | 结论 |
| --- | --- | --- | --- | ---: | --- | --- |
| E0 | Endpoint pilot | 50/task, stride 4, `t -> t+16` | 完成 625 steps | 0.76680 | 未作为正式 gate | 工程与吞吐验证通过 |
| E1 | Endpoint formal, batch 128 | 50/task, stride 4, `t -> t+16` | 1,875 steps early stop | 0.71436 | `t` 最近 100%，`t+16` 最近 0% | 优于 persistence，但没有推进到 endpoint |
| E2 | Endpoint batch-32 | 与 E1 sample exposure 匹配 | 4,000 steps early stop | 0.71223 | `t` 最近 100%，future 中 `t+4` 最近 100% | 减小 batch 不是解决方案 |
| E3 | Causal multi-time | 预测 `t+4/8/12/16`，prefix action mask | 5,000 steps early stop | 0.70265 (`t+16`) | `t` 最近 100%；future retrieval 到 `t+16` 1.03% | 辅助目标改善回归，但未消除 contraction |
| E4 | AC-normalized L1 | E3 + affine-free LN + L1 | 4,500 steps early stop | 0.79153* | `t` 最近 100%；future retrieval 到 `t+16` 1.05% | 正确 representation contract 也不能解决 contraction |
| S1 | Single-step 50/task | stride 4, `t -> t+4`, joint visual+proprio | 15 epochs early stop | 0.7657 | final train 70.07%，val 0% | 网络能拟合 train，但无法泛化 |
| S2 | Single-step 100/task，旧 encoder | stride 1, joint visual+proprio | epoch 16 手动停止 | 0.7337（epoch 4） | epoch 16 train 51.27%，val 0.26% | encoder 错配确认后停止；仍是明显过拟合 |
| N1 | Native-256 joint | 100/task, stride 1, `z+s+a -> z'+s'` | 完成 25 epochs | 0.72595（epoch 5） | final train 72.71%，val 2.21% | 真正 native-256 仍未通过泛化 gate |
| N2 | Native-256 visual-only target | 保留 `s_t` 输入，不预测 `s_{t+4}` | 完成 25 epochs | 0.72548（epoch 6） | final train 72.90%，val 2.46% | 删除 future proprio target 基本无影响 |
| N3 | Native-256 no-proprio | 仅 `z_t + 4 actions -> z_{t+4}` | **运行中，11/25 epochs** | 0.72564（epoch 5，暂定） | epoch 11 train 17.09%，val 0.32% | 前 11 epochs 与 N1/N2 基本重合，继续跑满协议 |

\* E4 改变了 latent geometry 和 loss，`0.79153` 不能与 E3 的 raw-latent normalized
error 作严格数值比较；可比较的 temporal retrieval 仍然失败。

## 5. Endpoint predictor 系列（历史诊断）

### 5.1 固定设定

- 四任务：`KettleBoiling`、`LoadDishwasher`、`PreSoakPan`、
  `RinseSinkBasin`；
- 50 train / 10 validation / 20 test episodes per task；
- 20 Hz，16 actions 对应 0.8 秒；
- 16 个有顺序的 12-D action tokens，一个 current 16-D proprio token；
- predictor：width/depth/heads = `960/7/12`；
- 目标为 native-grid 形式的一个或多个 future latent；
- 早期 sampler 是 15 个 `(task, subtask)` group 等权，不是严格四任务各 25%。

### 5.2 E1：直接 `t+16` baseline

正式 seed-0 run 在 step 1,875 early-stop，最佳 step 为 1,500。完整 validation：

| 指标 | 结果 |
| --- | ---: |
| val normalized error | 0.71436 |
| val delta cosine | 0.51839 |
| correct beats zero action | 93.37% |
| correct beats far same-task action | 82.64% |
| prediction 最近当前 `t` | 100.00% |
| 排除 `t` 后最近 `t+4` | 100.00% |
| prediction 最近 `t+16` | 0.00% |

最佳 checkpoint 曾在完整 10,897-window test 上评估一次，test normalized error 为
0.71420。这个模型确实利用动作并优于 persistence，但其预测只是从 `t` 向未来移动
了一小段，不能作为 16-step residual/CEM 的后果模型。

### 5.3 E2：减小 global batch

global batch 从 128 降到 32，optimizer steps 按样本曝光从 2,250 上限换算到 9,000。
run 在 step 4,000 early-stop，最佳 step 2,500。

- normalized error 由 0.71436 小幅改善到 0.71223；
- correct beats far action 从 82.64% 改善到 83.85%；
- 但所有 validation prediction 仍最近 `t`；排除 `t` 后仍全部最近 `t+4`；
- `t+16` retrieval 仍为 0%。

因此，大 global batch 或 update 数不足不是 temporal contraction 的主因。

### 5.4 E3：因果多时间监督

同一个 predictor 共享预测 `t+4/8/12/16`。每个 horizon 只允许看到对应 prefix action；
未来 action slots 通过 padding mask 隔离，autograd test 验证 `t+4` loss 对 action 5--16
的梯度严格为 0。

| Requested endpoint | Val normalized error | Delta cosine |
| --- | ---: | ---: |
| `t+4` | 0.75355 | 0.47090 |
| `t+8` | 0.72596 | 0.50226 |
| `t+12` | 0.71207 | 0.51871 |
| `t+16` | 0.70265 | 0.52964 |

未来时间 retrieval 从“全部 `t+4`”变为：`t+4` 91.27%、`t+8` 5.77%、`t+12`
1.93%、`t+16` 1.03%。但包含当前状态时仍有 100% 的 prediction 最近 `t`。这说明
multi-time supervision 能改善 endpoint regression，却仍不能学习足够大的真实时间推进。

### 5.5 E4：V-JEPA2-AC normalization + L1

该实验恢复 V-JEPA2-AC 风格的 affine-free per-token LayerNorm，并把 Huber 换为 L1。
预测 delta RMS ratio 从 56.55% 增到 63.33%，但投影到真实 future direction 的比例
几乎不变（30.86% -> 30.48%）。`t+16` future retrieval 也只从 1.03% 变为 1.05%。

结论是：正确的 AC representation/loss contract 应保留作参考，但它不是 contraction
的根因。继续扩大相同 direct-endpoint objective 的数据量没有充分依据。

## 6. Single-step `t -> t+4` 系列

### 6.1 为什么改成单步

研究框架在每个 subtask 开始时生成完整目标视频，后续可以逐段比较预测状态与视频
对应时刻。因此不必强迫一个网络直接跳到 `t+16`；先学习稳定的 4-control-step
transition（0.2 秒），以后再共享权重 rollout 到 `t+8/12/16` 更符合规划需求。

single-step 输入为四个独立、有顺序的 action tokens，而不是把四步或十六步 action
压进一个 token。视觉 context 只使用 `z_t`，没有引入 `z_{t-4}`，也没有把 recorded
future proprio 当输入。

### 6.2 S1：50/task、stride 4

该 run 使用旧的 384-weight/256-crop cache。模型在 15 epochs / 13,545 steps 后
early-stop，最佳 epoch 为 4：

| Epoch | Train norm | Train future_closer | Val norm | Val future_closer |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8709 | 0.00% | 0.8740 | 0.00% |
| 4（best） | 0.7417 | 0.00% | 0.7657 | 0.00% |
| 10 | 0.5819 | 21.68% | 0.8057 | 0.00% |
| 15 | 0.4282 | 70.07% | 0.8663 | 0.00% |

最佳 checkpoint 上 correct action 优于 zero/shuffle 的比例分别为 87.22%/77.75%。
所以问题不是完全忽略 action，而是 train dynamics 可以记住、held-out episode
无法泛化。future proprio head 也没有达到 rollout 质量：其 val state normalized error
为 22.46。

### 6.3 S2：100/task、stride 1，旧 encoder

训练扩展为 100 trajectories/task、227,408 train windows、stride 1、global batch 128、
固定 25 epochs。encoder 错配确认后在 epoch 16 手动停止：

- 最佳 val normalized error：0.7337（epoch 4）；
- epoch 16 train future_closer：51.27%，dynamic subset 72.20%；
- epoch 16 val future_closer：0.26%，dynamic subset 0.53%；
- epoch 16 val normalized error 已恶化到 0.8240。

该 run 同样表现为 train 拟合和 validation 失败，但因为 encoder 配置错误，不再继续
剩余九个 epochs，也不作为当前正式结果。

## 7. 原生 V-JEPA2-256 的受控三实验

三个 runs 共享以下设置：

- 100 train trajectories/task，validation 固定 10/task，test 锁定且未 cache/未读取；
- stride 1；227,408 train / 22,139 validation windows；
- camera：`robot0_agentview_left`；
- V-JEPA2 native `[16,16,1408]` feature grid；
- predictor `960 x 7 x 12`，约 80.5M 参数；
- global batch 128（64/GPU），两张 A40；
- AdamW，LR `3e-4`，weight decay `0.05`，5% warmup + cosine；
- 固定 25 epochs / 44,425 updates，不 early-stop；
- 每 epoch 完整 validation，checkpoint 只保存 epoch 5/10/15/20/25；
- seed 0，共享参数初始化在消融间逐元素一致；
- 四任务等权，task 内再平衡 subtask。

### 7.1 N1：视觉 + future proprio 联合预测

模型契约：

```text
inputs:  z_t, s_t, a_t, a_{t+1}, a_{t+2}, a_{t+3}
targets: z_{t+4}, s_{t+4}
loss:    L_visual + 0.005 * L_state
```

25 epochs 完成用时 169.54 分钟。最佳 validation 出现在 epoch 5，而 train 在后续继续
改善、validation 持续恶化：

| 指标 | Best epoch 5 | Final epoch 25 |
| --- | ---: | ---: |
| train visual normalized error | 0.68591 | 0.42948 |
| train future_closer | 0.00% | 72.71% |
| val visual normalized error | **0.72595** | 0.84546 |
| val future_closer | 0.00% | 2.21% |
| val dynamic future_closer | 0.00% | 4.59% |
| val correct beats shuffle | 88.53% | 83.98% |

final train dynamic future_closer 已达 91.31%，说明模型容量足以拟合训练窗口；但 held-out
validation 仍无法稳定跨过时间中点。state head 的 val normalized error 在最佳视觉 epoch
为 44.19，不能作为下一步 autoregressive state 使用。

### 7.2 N2：删除 future proprio target，保留 current proprio condition

模型契约：

```text
inputs: z_t, s_t, a_t, a_{t+1}, a_{t+2}, a_{t+3}
target: z_{t+4}
loss:   L_visual
```

25 epochs 完成用时 168.53 分钟：

| 指标 | Best epoch 6 | Final epoch 25 |
| --- | ---: | ---: |
| train visual normalized error | 0.66601 | 0.43172 |
| train future_closer | 0.34% | 72.90% |
| val visual normalized error | **0.72548** | 0.84388 |
| val future_closer | 0.014% | 2.46% |
| val dynamic future_closer | 0.029% | 5.05% |
| val correct beats shuffle | 89.38% | 84.48% |

N1 和 N2 的最佳 norm 只差 0.00047，完整学习轨迹也几乎重合。由此可以排除一个重要
怀疑：预测 `s_{t+4}` 的 auxiliary head 及其 `0.005` loss 并不是视觉泛化失败的主因。

### 7.3 N3：完全不使用 proprio（当前运行）

模型契约严格为：

```text
inputs: z_t, a_t, a_{t+1}, a_{t+2}, a_{t+3}
target: z_{t+4}
```

模型没有 current-state token、future-state head、proprio loss 或 state metrics，参数量
80,476,224。它与 N2 的唯一关键差别是删除 `s_t` condition，用来判断 current proprio
是否帮助模型记忆训练轨迹或形成 shortcut。

运行状态（截至 2026-08-04 12:39 CEST）：

- 2026-08-04 11:21:39 启动；
- 227,408 train / 22,139 validation windows，1,777 steps/epoch；
- 已完成 11/25 epochs 和 19,547/44,425 optimizer steps（44%），epoch 12 正在运行；
- 两张 A40 utilization 为 100%/99%，显存均约 7.15 GiB；
- 上游 N2 completion validation 通过，`upstream_status=complete`；
- epoch 5 和 epoch 10 checkpoint 已正常保存；
- 按当前每 epoch 约 6.71 分钟的速度，预计 14:10 CEST 左右完成。

当前学习曲线：

| Epoch | Global step | Train norm | Train future_closer | Val norm | Val future_closer | Val dynamic future_closer | Val correct beats shuffle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,777 | 0.84227 | 0.00% | 0.84496 | 0.00% | 0.00% | 77.37% |
| 5 | 8,885 | 0.68596 | 0.00% | **0.72564** | 0.00% | 0.00% | 89.16% |
| 10 | 17,770 | 0.59866 | 12.16% | 0.74802 | 0.44% | 0.92% | 87.81% |
| 11 | 19,547 | 0.58267 | 17.09% | 0.75402 | 0.32% | 0.66% | 87.94% |

截至 epoch 11，最佳 validation normalized error 出现在 epoch 5。之后 train norm 和
train future_closer 持续改善，而 validation norm 已开始回升，validation
future_closer 仍低于 0.5%。它已经表现出与 N1/N2 相同的 train 拟合、validation
时间推进不足趋势。

三组实验在相同 epoch 11 的直接对照如下：

| 实验 | Train norm | Train future_closer | Val norm | Val future_closer | Val dynamic future_closer | Val correct beats shuffle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N1：联合预测 visual + proprio | 0.58053 | 17.38% | 0.75398 | 0.42% | 0.88% | 87.40% |
| N2：只预测 visual，输入 current proprio | 0.58132 | 18.21% | 0.75548 | 0.44% | 0.93% | 88.06% |
| N3：只预测 visual，完全无 proprio | 0.58267 | 17.09% | 0.75402 | 0.32% | 0.66% | 87.94% |

三条曲线到 epoch 11 几乎重合。当前证据说明删除 current proprio 尚未改善泛化，也没有
破坏 action sensitivity；最终判断以 N3 完成 25 epochs 后的同协议比较为准。

第一次自动 launch 曾因 `pgrep -f` 把 tmux server 的历史 command line 误判为残留
trainer 而退出；当时没有产生 history 或 checkpoint。idle detector 已改为扫描
`/proc`，只接受真实 Python executable 且 argv 明确包含训练脚本的进程，随后正常重启。
该队列故障不改变实验数据或训练逻辑。

## 8. Supporting runs 与未形成科学结论的尝试

### 8.1 Gradient audits 和 DDP smoke

- single-step 最初 proposed proprio weight `0.1` 会产生约 3.84 倍于视觉梯度的
  weighted shared-trunk gradient，因此改为 `0.005`；正式 audit ratio 约 0.12--0.17。
- native-256 global-batch-128 smoke 完成 20 updates，双卡、fused AdamW、validation
  和 checkpoint path 均通过；peak CUDA allocation 为 5.84 GiB/rank。
- targeted/full regression tests 在 no-proprio implementation 后为 11/11 和 43/43。

这些结果证明实现和资源配置可运行，不应当作模型泛化结果。

### 8.2 Frozen-encoder dynamics bake-off smoke

曾搭建 GR00T Eagle2 与 V-JEPA2 的同协议 multi-horizon bake-off，并完成小型 GR00T
smoke（29,488 参数）。formal three-seed encoder comparison 没有完成，因此该 smoke
的 normalized error 9.36 不具有科学解释价值，也不应与 80M predictor runs 比较。

### 8.3 JEPA-WMs RoboCasa stage-1 reproduction

该 reproduction 使用作者的 RoboCasa 0.2.0 fork、自定义 `PnPCounterTop` task，以及
DROID 训练的 V-JEPA2-AC world model，不是 RoboCasa365 训练实验。

- 一条 2-candidate/2-iteration CEM quick-debug 完成，结果 0/1，设置过小而不可诊断；
- paper-sized 32-episode two-GPU run 成功启动，但一次 300-candidate、15-iteration CEM
  已超过 5 分钟/A40，因成本过高在首个 action plan 完成前主动停止；
- 没有产生 formal success-rate，不能声称复现论文结果。

该阶段的 payload、环境和旧 assets 后来为释放空间而删除，只保留文档记录。

## 9. 当前可以下的结论

1. **direct `t+16` regression 不够。** normalized error 约 0.70 看起来不错，但 temporal
   retrieval 证明模型仍停留在 `t` 附近；这个 checkpoint 不适合 residual/CEM。
2. **减小 batch 不是答案。** 样本曝光匹配后 endpoint norm 略好，但时间位置完全没变。
3. **multi-time auxiliary supervision 有帮助但不充分。** 它让 future-only retrieval
   出现少量 `t+8/12/16`，却不能让 prediction 越过当前状态中点。
4. **AC normalization/L1 不是缺失的关键。** 它增加预测运动幅度，没有改善正确的未来
   方向投影。
5. **single-step 问题主要是泛化，不是 train optimization。** 多个 runs 都能让 train
   future_closer 达到约 70%，validation 却只到 0--2.5%。
6. **模型会使用 action。** correct-vs-zero/shuffle 通常明显高于随机，因此不能简单归因
   为 action token 完全失效；但 action sensitivity 仍不足以证明 learned dynamics 可用。
7. **删除 future proprio target 无实质影响。** N1/N2 几乎相同，辅助 state head 不是
   当前视觉失败的原因。
8. **current proprio input 暂未表现为主要 shortcut。** N3 到 epoch 11 与 N1/N2 的
   train/validation 曲线和 action sensitivity 基本重合；完整结论仍以 epoch 25 为准。

## 10. N3 完成后的决策规则

比较 N2 与 N3 时，优先看同 epoch 和最佳 validation 的：

1. overall/dynamic `future_closer`；
2. visual normalized error 与 train-validation gap；
3. delta cosine 和 predicted/true delta RMS ratio；
4. correct action versus zero/shuffle；
5. 四任务是否一致，而非单一任务拉动平均值。

可能的解释：

- 如果 N3 validation future_closer 明显提高，同时 action sensitivity 保持，说明 `s_t`
  可能是训练 shortcut，应继续纯视觉条件或重新设计 proprio conditioning。
- 如果 N3 与 N2 仍近似，proprio 不是主因；优先检查单视角表征、episode-level视觉
  泛化、action/state 时间对齐和训练目标，而不是继续删除 token。
- 如果 N3 明显更差，说明 current proprio 提供必要的机器人构型信息；应保留它，但需
  通过更多跨 episode 数据、多视角/更合适 encoder 或正则化改善泛化。

在 single-step validation 明显通过 future-closeness gate 之前，仍不启动 two-step
self-rollout、residual policy 或 CEM 正式训练。

## 11. Artifact 状态

当前保留：

- native-256 feature cache；
- N1、N2 的 `history.json`、`summary.json`、日志和 epoch 5/10/15/20/25 checkpoints；
- N3 的 live log、queue context，以及 epoch 5/10 checkpoints，后续按相同周期保存；
- 本文以及两个详细实验协议文档。

为释放磁盘已永久删除：

- 早期 endpoint formal runs 的 checkpoints/cache artifacts（数值记录仍在文档）；
- 50/task single-step formal checkpoints；
- superseded 384-weight/256-crop 的 100/task cache 和 stopped-run checkpoints；
- JEPA-WMs reproduction payload/environment/旧 assets；
- 与当前研究无关的旧 model/data payload。

主要结果路径：

```text
outputs/single_step_dynamics/formal_native256_100_per_task_stride1_b128_seed_0/
outputs/single_step_dynamics/formal_native256_visual_only_100_per_task_stride1_b128_seed_0/
outputs/single_step_dynamics/formal_native256_no_proprio_100_per_task_stride1_b128_seed_0/
```

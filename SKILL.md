---
name: grpo-train-empty-run
description: Diagnose and fix a GRPO/GEPO RL training run that produces no learning signal — loss stuck at 0.0, grad_norm at 0.0, or frac_reward_zero_std at 1.0 (group rewards always identical). Trigger on "训练没在学 / 空转 / loss 恒为 0 / grad_norm 为 0 / 奖励全相同 / no gradient / training not learning" in any GRPO/GEPO/TRL-based RL training project (Hetero-RL, TRL, HF GRPOTrainer). Also covers the adjacent failure modes seen on the same setup: full-model OOM at optimizer.step, completions never terminating, stdout-buffered metrics not appearing in logs, and eval batch not divisible by num_generations.
category: ml_training
version: 1.1
argument-hint: "<training log path or symptom>"
---

# GRPO/GEPO 训练空转排查与修复

Investigate: **$ARGUMENTS**

RL 训练"跑了很久但没在学"是最隐蔽的问题：loss/grad 都正常显示 0，训练照常走完，checkpoint 照常保存，但模型权重几乎没动。本 skill 是实际排障过程的标准化沉淀，按步骤走完即可定位根因并修复。

## 何时触发

- **正向**：用户报告 GRPO/GEPO/TRL 训练"没在学"、loss 恒 0、grad_norm 恒 0、奖励全相同；或提供 trainer_state.json/训练日志要求排查
- **禁止调用**：非 RL 训练问题、纯 SFT/DPO 损失异常（除非明确是 RL 空转特征）

## 第一步：运行诊断器（30秒定位）

```bash
python3 ~/.openclaw/skills/grpo-train-empty-run/scripts/check_empty_run.py <trainer_state.json 或 日志 或 目录>
```

诊断器输出 verdict：`EMPTY_RUN_CONFIRMED` / `CHECK_LORA` / `CHECK_REWARD` / `DEGRADING` / `HEALTHY`，并给出证据。

无日志文件时，按下方「症状识别」人工核对。

## 症状识别（先确认是不是这个病）

查看 `trainer_state.json` 的 `log_history` 或训练日志的指标行：

| 指标 | 空转特征 | 健康特征 |
|---|---|---|
| `loss` | 恒为 `0.0` | 非零（或随 batch 有起伏） |
| `grad_norm` | 恒为 `0.0` | > 0，且连续多步不退化 |
| `frac_reward_zero_std` | 恒为 `1.0` | < 1.0（组内奖励有方差） |
| `reward_std` / `rewards/*/std` | `0.0` | > 0 |
| `adv_std` | `nan`（张量只剩 1 个元素） | 非 nan |

**核心判据**：`frac_reward_zero_std == 1.0` 且 `grad_norm == 0.0` 连续多步 → 空转确诊。
注意陷阱：空转运行的"完成"很具迷惑性——120 步 7 小时跑完、checkpoint 齐全、日志末尾 `finished!`，但全程没学。

## 根因链（按出现频率排序，逐一排查）

### 根因 A：离散奖励函数 → 组内奖励打平（最常见）
奖励函数只返回少量离散值（如 `{0, 0.3, 1.0}`），且 `num_generations` 太小（如 2）。
→ 同 prompt 的 G 条 completion 几乎必然落入同一奖励桶 → `std_grouped_rewards = 0` → `advantages = rewards - mean = 0` → loss 恒 0。
**判定**：日志里 `rewards/*/mean` 恒定、`rewards/*/std = 0.0`，且奖励函数实现里是 `if/else` 返回固定值。

### 根因 B：LoRA/PeFT 未启用 → 全量参数训练 OOM
config（yaml/CLI）漏配 `use_peft` / `lora_*` → 全量 9.37B 参数可训练 → `optimizer.step()` 初始化 Adam 状态时 OOM 崩溃。
**判定**：日志 `Number of trainable parameters` 显示 ~9B（全量）而非 ~50M（LoRA）；崩溃点在 `optimizer.step()` 的 `torch.OutOfMemoryError`。

### 根因 C：completion 永不终止 → 奖励提取失败
`max_completion_length` 过大 + 模型不输出 EOS → 所有 completion 被截断（`clipped_ratio = 1.0`、`mean_terminated_length = 0`）→ 奖励函数从截断的垃圾文本里提取不到有效答案。
**判定**：指标行 `completions/clipped_ratio = 1.0`、`completions/mean_terminated_length = 0`。

### 根因 D：同配置链上的环境/工程问题（常同时出现）
1. **eval batch 不能被 num_generations 整除** → 启动即报错 `global eval batch size must be evenly divisible by num_generations` → 把 `per_device_eval_batch_size` 提到 num_generations 的倍数。
2. **指标不进日志** → stdout 重定向到文件时 Python 默认块缓冲，指标 dict 被吞 → 启动脚本加 `export PYTHONUNBUFFERED=1`。
3. **Python 环境缺依赖**（如 `latex2sympy2_extended`）→ 报 `ModuleNotFoundError` → 显式指定依赖齐全的 conda 环境。

## 修复步骤（按序执行，每步后验证）

1. **奖励函数连续化**（治根因 A）：
   把离散分值改为按误差连续递减。例：错误答案从固定 `0.3` 改为
   `0.3 * max(0.0, 1.0 - min(rel_err, 1.0))`，其中 `rel_err = abs(pred - gold) / max(abs(gold), 1.0)`。
   阈值放宽到 100% 误差才归零，避免"大部分错误答案又被压成 0"。
   **单测**：喂 8 条不同 completion（答案各异），确认 `np.std(rewards) > 0`。

2. **增大 num_generations**（治根因 A）：2 → 8（组内奖励不再打平）。
   同步把 `per_device_eval_batch_size` 调到能被 num_generations 整除（如 8）。

3. **启用 LoRA**（治根因 B）：config 补齐 `use_peft: true`、`lora_r/alpha/dropout`、
   `lora_target_modules`（q/k/v/o/gate/up/down_proj）。确认日志可训参数降到 ~50M。

4. **收紧 max_completion_length**（治根因 C）：1024 → 512，让 completion 更可能正常终止，
   同时生成提速约 4 倍。

5. **环境与日志**（治根因 D）：启动脚本加 `export PYTHONUNBUFFERED=1`；
   显式指定依赖齐全的 conda 环境路径。

## 验证标准（修复后必须全部满足）

重启训练，等首步（或前 3 步）指标出现：

- [ ] `grad_norm > 0`，且连续 3 步不退化（旧空转运行第 1 步偶有非零、之后退化为 0）
- [ ] `frac_reward_zero_std < 1.0`（最好 = 0.0）
- [ ] `reward_std > 0`（组内奖励有方差）
- [ ] 无 OOM 崩溃，训练能连续跑过多步
- [ ] （可选）指标行实时出现在日志里（PYTHONUNBUFFERED 生效）

对比基准：修复前 grad_norm 恒 0、frac_reward_zero_std 恒 1；修复后两项同时翻转即确认生效。

## 注意事项

- **别被"跑完了"骗了**：空转运行也能正常走完 120 步并保存 checkpoint。必须看指标，不是看完成。
- **先确认再修**：每一条根因都有对应的日志/指标证据，找到证据再动手，不要猜。
- **损失显示 0.0 但 grad_norm > 0 是正常现象**（GRPO/GEPO 优势均值归零所致），以 grad_norm 为准。
- **连续奖励的阈值要放宽**：若阈值过严（如误差 ≥10% 就归 0），错误答案又会被压成同一分值，空转复发——这正是本排障中踩过的坑。
- 一次只改一处并验证，避免多根因叠加时无法归因。

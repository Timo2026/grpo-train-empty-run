<!--
  作者: timo.cao | 邮箱: miscdd@163.com
  生成: 大帅教练系统 (dashuai coach)
  许可: MIT License
-->

---
name: grpo-train-empty-run
description: Diagnose and fix a GRPO/GEPO RL training run that produces no learning signal — loss stuck at 0.0, grad_norm at 0.0, or frac_reward_zero_std at 1.0 (group rewards always identical). Trigger on "训练没在学 / 空转 / loss 恒为 0 / grad_norm 为 0 / 奖励全相同 / no gradient / training not learning" in any GRPO/GEPO/TRL-based RL training project (Hetero-RL, TRL, HF GRPOTrainer). Also covers adjacent failure modes: full-model OOM at optimizer.step, completions never terminating, stdout-buffered metrics missing from logs, eval batch not divisible by num_generations.
category: ml_training
version: 1.0
argument-hint: "<training log path or symptom>"
---

# GRPO/GEPO 训练空转排查与修复

RL训练"跑了很久但没在学"是最隐蔽的问题：loss/grad都正常显示0，训练照常走完，checkpoint照常保存，但模型权重几乎没动。本skill是实际排障过程的标准化沉淀。

## 何时触发

- **正向**：用户报告GRPO/GEPO/TRL训练"没在学"、loss恒0、grad_norm恒0、奖励全相同；或提供trainer_state.json/训练日志要求排查
- **禁止调用**：非RL训练问题、纯SFT/DPO损失异常（除非明确是RL空转特征）

## 第一步：运行诊断器（30秒定位）

```bash
python3 ~/.openclaw/skills/grpo-train-empty-run/scripts/check_empty_run.py <trainer_state.json 或 日志 或 目录>
```

诊断器输出 verdict：`EMPTY_RUN_CONFIRMED` / `CHECK_LORA` / `CHECK_REWARD` / `DEGRADING` / `HEALTHY`，并给出证据。

无日志文件时，按下方「症状识别」人工核对。

## 症状识别

| 指标 | 空转特征 | 健康特征 |
|---|---|---|
| `loss` | 恒为 `0.0` | 非零或起伏 |
| `grad_norm` | 恒为 `0.0` | > 0，连续多步不退化 |
| `frac_reward_zero_std` | 恒为 `1.0` | < 1.0 |
| `reward_std` | `0.0` | > 0 |

**核心判据**：`frac_reward_zero_std == 1.0` 且 `grad_norm == 0.0` 连续多步 → 空转确诊。
陷阱：空转的"完成"很迷惑——120步7小时跑完、checkpoint齐全、`finished!`，但全程没学。

## 根因链（按频率排查）

### A. 离散奖励 → 组内打平（最常见）
奖励只返回少量离散值（`{0,0.3,1.0}`）+ `num_generations`太小（如2）→ 组内奖励必然打平 → advantages=0 → loss恒0。
**修复**：
```bash
# 奖励连续化 (错误答案按误差连续递减, 阈值放宽到100%误差才归零)
python3 ~/.openclaw/skills/grpo-train-empty-run/scripts/fix_reward.py --test   # 单测: 确认std>0
```
同时 num_generations 2→8，`per_device_eval_batch_size` 调到能被其整除。

### B. LoRA未启用 → 全量OOM
config漏配 `use_peft`/`lora_*` → 全量9.37B可训练 → optimizer.step() OOM。
**判定**：日志`Number of trainable parameters` ~9B（应~50M）。
**修复**：补 use_peft: true、lora_r/alpha/dropout、lora_target_modules (q/k/v/o/gate/up/down_proj)。

### C. completion永不终止 → 奖励提取失败
`max_completion_length`过大 + 模型不输出EOS → 全被截断（`clipped_ratio=1.0`、`mean_terminated_length=0`）→ 奖励从垃圾文本提取不到。
**修复**：max_completion_length 1024→512（生成也快~4倍）。

### D. 环境工程（常同时出现）
1. eval batch不能整除num_generations → 启动报错 → batch调到倍数
2. 指标不进日志 → `export PYTHONUNBUFFERED=1`
3. 缺依赖（latex2sympy2_extended等）→ 显式指定conda环境

## 验证标准（修复后必须全部满足）

重启训练，等首步或前3步指标：
- [ ] `grad_norm > 0` 连续3步不退化（旧空转第1步偶非零后退化0）
- [ ] `frac_reward_zero_std < 1.0`（最好0.0）
- [ ] `reward_std > 0`
- [ ] 无OOM，连续跑过多步
- [ ] （可选）指标实时进日志（PYTHONUNBUFFERED生效）

对比基准：修复前 grad_norm恒0+frac恒1 → 修复后两项同时翻转即生效。

## 注意事项

- **别被"跑完了"骗了**：必须看指标，不是看完成
- **先确认再修**：每条根因都有日志证据，找到证据再动手
- **loss=0但grad_norm>0是正常**（优势均值归零），以grad_norm为准
- **连续奖励阈值放宽到100%**：过严（10%归0）会导致空转复发
- **一次只改一处**，避免多根因叠加无法归因

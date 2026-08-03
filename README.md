<!--
  作者: timo.cao | 邮箱: miscdd@163.com
  生成: 大帅教练系统 (dashuai coach)
  许可: MIT License
-->

# GRPO Train Empty-Run Diagnoser

[中文](#中文) | [English](#english)

---

## 中文

### 这是什么？

GRPO/GEPO（RL训练）"空转"排查与修复工具集。当你的强化学习训练**跑了很久但没在学**——loss 恒为 0.0、grad_norm 恒为 0.0、组内奖励完全相同（frac_reward_zero_std=1.0）——用它 30 秒定位根因。

这是实际排障过程的标准化沉淀，覆盖 TRL / HF GRPOTrainer / Hetero-RL 等 GRPO 系框架。

### 空转的症状

| 指标 | 空转特征 | 健康特征 |
|---|---|---|
| `loss` | 恒为 `0.0` | 非零或起伏 |
| `grad_norm` | 恒为 `0.0` | > 0，连续多步不退化 |
| `frac_reward_zero_std` | 恒为 `1.0` | < 1.0 |
| `reward_std` | `0.0` | > 0 |

⚠️ **陷阱**：空转运行也能正常跑完 120 步、保存 checkpoint、日志末尾 `finished!`——必须看指标，不是看完成。

### 快速使用

```bash
# 1. 诊断：传入 trainer_state.json / 日志 / 目录
python3 scripts/check_empty_run.py <trainer_state.json>

# 2. 奖励连续化单测（根因A修复）
python3 scripts/fix_reward.py --test
```

### 四大根因

| 根因 | 症状 | 修复 |
|---|---|---|
| **A. 离散奖励打平**（最常见） | frac_reward_zero_std=1.0 + grad_norm=0.0 | 奖励连续化 + num_generations 2→8 |
| **B. LoRA未启用** | optimizer.step() 处 OOM | 补 use_peft/lora_* 配置 |
| **C. completion不终止** | clipped_ratio=1.0 + mean_terminated_length=0 | max_completion_length 1024→512 |
| **D. 环境工程** | 启动报错/指标丢失 | batch整除 + PYTHONUNBUFFERED + 依赖 |

详细速查表见 `references/root_causes.md`。

---

## English

### What is this?

A diagnosis & fix toolkit for **GRPO/GEPO RL training "empty runs"** — training that runs for hours but learns nothing: loss stuck at 0.0, grad_norm stuck at 0.0, group rewards always identical (frac_reward_zero_std=1.0).

Distilled from real debugging sessions. Works with TRL / HF GRPOTrainer / Hetero-RL and similar GRPO-family frameworks.

### Symptoms

| Metric | Empty-run | Healthy |
|---|---|---|
| `loss` | always `0.0` | non-zero / varies |
| `grad_norm` | always `0.0` | > 0, sustained |
| `frac_reward_zero_std` | always `1.0` | < 1.0 |
| `reward_std` | `0.0` | > 0 |

⚠️ **Gotcha**: an empty run can still complete 120 steps, save checkpoints, and print `finished!` — always check metrics, not completion.

### Quick Start

```bash
# 1. Diagnose: pass trainer_state.json / log file / directory
python3 scripts/check_empty_run.py <trainer_state.json>

# 2. Reward-continuity unit test (root cause A fix)
python3 scripts/fix_reward.py --test
```

### Four Root Causes

| Cause | Symptom | Fix |
|---|---|---|
| **A. Discrete reward flattening** (most common) | frac_reward_zero_std=1.0 + grad_norm=0.0 | continuous reward + num_generations 2→8 |
| **B. LoRA not enabled** | OOM at optimizer.step() | add use_peft/lora_* config |
| **C. Completions never terminate** | clipped_ratio=1.0 + mean_terminated_length=0 | max_completion_length 1024→512 |
| **D. Environment issues** | startup error / missing metrics | batch divisibility + PYTHONUNBUFFERED + deps |

See `references/root_causes.md` for the full cheatsheet.

---

## License

MIT License

# GRPO/GEPO 空转 · 根因速查表

## 症状 → 根因 → 修复 一页速查

| 症状 (日志指标) | 根因 | 修复 | 验证 |
|---|---|---|---|
| `frac_reward_zero_std=1.0` + `grad_norm=0.0` | **A. 离散奖励打平** | 奖励连续化 + num_generations 2→8 | std>0, frac<1.0 |
| `optimizer.step()` 处 OOM | **B. LoRA未启用** | 补 use_peft/lora_* 配置 | 可训参数 ~50M 而非 ~9B |
| `clipped_ratio=1.0` + `mean_terminated_length=0` | **C. completion不终止** | max_completion_length 1024→512 | mean_terminated_length>0 |
| 启动报 `eval batch not divisible by num_generations` | **D1. eval batch整除** | per_device_eval_batch_size 调为 num_generations 倍数 | 启动通过 |
| 指标不进日志 | **D2. stdout块缓冲** | `export PYTHONUNBUFFERED=1` | 指标实时出现 |
| `ModuleNotFoundError: latex2sympy2_extended` | **D3. 缺依赖** | 显式指定依赖齐全的conda环境 | 导入通过 |

## 关键判据

- **核心确诊**: `frac_reward_zero_std == 1.0` 且 `grad_norm == 0.0` 连续多步
- **陷阱**: 空转也能正常跑完120步、保存checkpoint、日志末尾`finished!`——必须看指标，不看完成
- **正常现象**: loss显示0.0但grad_norm>0（GRPO优势均值归零），以grad_norm为准
- **复发坑**: 连续奖励阈值过严（误差≥10%归0）→ 错误答案又被压成同一分值 → 空转复发。阈值放宽到100%

## 检查清单 (修复后)

- [ ] `grad_norm > 0` 连续3步不退化
- [ ] `frac_reward_zero_std < 1.0` (最好=0.0)
- [ ] `reward_std > 0`
- [ ] 无OOM，连续跑过多步
- [ ] 指标实时出现在日志

## 经验教训 (实际排障沉淀)

1. 一次只改一处并验证，避免多根因叠加无法归因
2. 先找证据再动手——每条根因都有对应日志指标，不要猜
3. 空转运行"完成"极具迷惑性，120步7小时跑完≠在学
4. 离散奖励+小num_generations是最常见组合拳，优先查

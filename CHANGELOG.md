<!--
  作者: timo.cao | 邮箱: miscdd@163.com
  生成: 大帅教练系统 (dashuai coach)
  许可: MIT License
-->

# Changelog

## [1.0.0] - 2026-08-03

### Added
- `scripts/check_empty_run.py`: GRPO/GEPO 空转诊断器
  - 解析 `trainer_state.json` 的 `log_history`，判定 `EMPTY_RUN_CONFIRMED` / `CHECK_LORA` / `CHECK_REWARD` / `DEGRADING` / `HEALTHY`
  - 支持目录扫描（自动找最新 trainer_state.json）与普通日志解析
  - 输出核心判据证据：loss/grad_norm/frac_reward_zero_std/reward_std 序列
- `scripts/fix_reward.py`: 奖励函数连续化工具
  - 离散奖励 → 按误差连续递减（阈值放宽到 100% 误差才归零）
  - 内置单测：8 条不同 completion 验证组内奖励 std > 0
- `references/root_causes.md`: 症状→根因→修复 一页速查表
- `SKILL.md`: 标准 AgentSkills 规范（frontmatter + 触发条件 + 排查流程 + 验证标准）

### 修复经验沉淀
- 根因 A（离散奖励打平）是最常见空转原因
- 连续奖励阈值过严会导致空转复发（≥10% 归零 → 错误答案又被压成同一分值）
- 空转运行"完成"极具迷惑性：120 步 7 小时跑完 ≠ 在学

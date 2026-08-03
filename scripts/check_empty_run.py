#!/usr/bin/env python3
"""GRPO/GEPO 训练空转诊断器
用法:
  python check_empty_run.py <trainer_state.json 或日志文件>
  python check_empty_run.py --dir <日志目录>   # 自动找最新 trainer_state.json

输出: 诊断结论 + 根因证据 (JSON)
"""
import json, sys, os, re, glob

def find_trainer_state(d):
    """在目录中找最新的 trainer_state.json"""
    cands = glob.glob(os.path.join(d, '**', 'trainer_state.json'), recursive=True)
    if not cands:
        cands = glob.glob(os.path.join(d, '**', '*state*.json'), recursive=True)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)

def parse_log(path):
    """解析普通日志文件中的指标行"""
    metrics = []
    patterns = [
        re.compile(r"'loss':\s*([\d.eE+-]+)"),
        re.compile(r"grad_norm[:\s]+([\d.eE+-]+)"),
        re.compile(r"frac_reward_zero_std[:\s]+([\d.eE+-]+)"),
        re.compile(r"reward_std[:\s]+([\d.eE+-]+)"),
    ]
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            row = {}
            if 'loss' in line or 'grad_norm' in line or 'reward' in line:
                for p in patterns:
                    m = p.search(line)
                    if m:
                        row[p.pattern[:20]] = float(m.group(1))
                if row:
                    metrics.append(row)
    return metrics

def analyze(trainer_state):
    """分析 trainer_state.json 的 log_history"""
    log_history = trainer_state.get('log_history', [])
    steps = []
    for entry in log_history:
        if 'train_runtime' in entry or 'epoch' in entry:
            continue
        steps.append(entry)

    if not steps:
        return {'verdict': 'UNKNOWN', 'reason': 'log_history 为空或格式不符', 'evidence': {}}

    # 提取关键指标序列
    loss_seq = [s.get('loss') for s in steps if s.get('loss') is not None]
    grad_seq = [s.get('grad_norm') for s in steps if s.get('grad_norm') is not None]
    frac_seq = [s.get('frac_reward_zero_std') for s in steps if s.get('frac_reward_zero_std') is not None]
    rstd_seq = []
    for s in steps:
        rw = s.get('rewards', {})
        if isinstance(rw, dict):
            std = rw.get('std')
            if std is not None:
                rstd_seq.append(std)

    evidence = {
        'total_steps': len(steps),
        'loss_nonzero': sum(1 for v in loss_seq if v != 0.0),
        'loss_all_zero': bool(loss_seq) and all(v == 0.0 for v in loss_seq),
        'grad_nonzero': sum(1 for v in grad_seq if v != 0.0),
        'grad_all_zero': bool(grad_seq) and all(v == 0.0 for v in grad_seq),
        'frac_all_one': bool(frac_seq) and all(v >= 1.0 for v in frac_seq),
        'reward_std_all_zero': bool(rstd_seq) and all(v == 0.0 for v in rstd_seq),
        'grad_tail': grad_seq[-3:] if grad_seq else [],
        'frac_tail': frac_seq[-3:] if frac_seq else [],
    }

    # 诊断
    if evidence['frac_all_one'] and evidence['grad_all_zero']:
        verdict = 'EMPTY_RUN_CONFIRMED'
        root_causes = []
        if evidence['reward_std_all_zero']:
            root_causes.append('A: 离散奖励+num_generations过小 → 组内奖励打平')
        if evidence['loss_all_zero']:
            root_causes.append('A/B: 奖励恒同(advantages=0) 或 全量参数OOM后无梯度')
        reason = '; '.join(root_causes) if root_causes else '核心判据命中: frac_reward_zero_std=1 且 grad_norm=0'
    elif evidence['grad_all_zero'] and not evidence['frac_all_one']:
        verdict = 'CHECK_LORA'
        reason = 'grad_norm恒0但组奖励有方差 → 检查use_peft/LoRA是否启用、学习率是否过小'
    elif evidence['frac_all_one'] and not evidence['grad_all_zero']:
        verdict = 'CHECK_REWARD'
        reason = '组奖励恒同但梯度非0 → 检查奖励函数离散化(放宽阈值)与num_generations'
    elif any(v == 0.0 for v in evidence['grad_tail']):
        verdict = 'DEGRADING'
        reason = f'梯度退化: 尾部grad_norm={evidence["grad_tail"]} → 奖励信号在衰减, 检查奖励连续性'
    else:
        verdict = 'HEALTHY'
        reason = f'训练正常: grad_norm尾部={evidence["grad_tail"]}, frac尾部={evidence["frac_tail"]}'

    return {'verdict': verdict, 'reason': reason, 'evidence': evidence}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': '用法: check_empty_run.py <trainer_state.json|日志|目录>'}, ensure_ascii=False))
        sys.exit(1)
    target = sys.argv[1]

    if os.path.isdir(target):
        ts_path = find_trainer_state(target)
        if not ts_path:
            # 尝试解析日志
            logs = glob.glob(os.path.join(target, '*.log')) + glob.glob(os.path.join(target, '*.txt'))
            if logs:
                metrics = parse_log(logs[-1])
                print(json.dumps({'verdict': 'LOG_ONLY', 'metrics_rows': len(metrics), 'note': '非trainer_state, 请人工核对指标'}, ensure_ascii=False))
            else:
                print(json.dumps({'error': f'目录 {target} 中未找到 trainer_state.json 或日志'}, ensure_ascii=False))
            sys.exit(0)
    else:
        ts_path = target

    try:
        if ts_path.endswith('.json'):
            with open(ts_path, encoding='utf-8') as f:
                data = json.load(f)
            if 'log_history' in data:
                result = analyze(data)
            else:
                metrics = parse_log(ts_path)
                result = {'verdict': 'LOG_ONLY', 'metrics_rows': len(metrics)}
        else:
            metrics = parse_log(ts_path)
            result = {'verdict': 'LOG_ONLY', 'metrics_rows': len(metrics)}
    except Exception as e:
        result = {'error': str(e)}

    result['source'] = ts_path
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

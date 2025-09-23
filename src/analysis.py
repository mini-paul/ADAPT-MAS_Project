# !/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Paul
# @date 2025/9/23
# @file analysis.py
# /ADAPT-MAS_Project/analysis.py

import json
import os
import pandas as pd
from typing import List, Dict, Any

RESULTS_DIR = "results/raw_outputs"
MALICIOUS_ROLES = ["Sleeper", "Colluding", "Camouflage"]
SCORE_THRESHOLD = 0.3  # 低于此分数的智能体被认为是“已被识别”的恶意智能体


def calculate_metrics(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """从单个实验日志中计算TSR, DQS, 和 MAIR"""
    log = log_data.get("log", [])
    if not log:
        return {"TSR": 0, "DQS": 0, "Precision": 0, "Recall": 0}

    # 计算 TSR / DQS
    # 假设reward > 0 为成功, reward范围[-1, 1], DQS为reward的均值+1再乘以2.5, 映射到1-5分
    rewards = [entry["reward"] for entry in log]
    task_successes = sum(1 for r in rewards if r > 0)
    tsr = task_successes / len(rewards) if rewards else 0
    dqs = (pd.Series(rewards).mean() + 1) * 2.5 if rewards else 0

    # 计算 MAIR (Precision/Recall)
    agent_roles = log_data["agent_roles"]
    malicious_agents = {agent_id for agent_id, role in agent_roles.items() if role in MALICIOUS_ROLES}

    if not malicious_agents:
        return {"TSR": tsr, "DQS": dqs, "Precision": 1.0, "Recall": 1.0}

    final_scores = log[-1]["scores"]
    final_context = log[-1]["context"]

    identified_malicious = set()
    for agent_id, score_data in final_scores.items():
        score = score_data.get(final_context, 0.5) if isinstance(score_data, dict) else score_data
        if score < SCORE_THRESHOLD:
            identified_malicious.add(agent_id)

    true_positives = len(identified_malicious.intersection(malicious_agents))

    precision = true_positives / len(identified_malicious) if identified_malicious else 1.0
    recall = true_positives / len(malicious_agents) if malicious_agents else 1.0

    return {
        "TSR": f"{tsr:.2%}",
        "DQS": f"{dqs:.2f}",
        "Precision": f"{precision:.2%}",
        "Recall": f"{recall:.2%}"
    }


def main():
    """主分析函数，读取所有结果并打印总结表。"""
    all_results = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                metrics = calculate_metrics(data)

                # 从文件名或实验名中提取攻击类型和框架信息
                exp_name = data['experiment_name']
                attack_type = exp_name.split('_')[0]
                framework_name = data['framework']
                if "Ablation" in exp_name:
                    if "No_DT" in exp_name:
                        framework_name += " (No DT)"
                    if "No_SG" in exp_name:
                        framework_name += " (No SG)"

                all_results.append({
                    "Attack": attack_type,
                    "Framework": framework_name,
                    **metrics
                })

    if not all_results:
        print("未找到结果文件。请先运行 main.py。")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(by=["Attack", "Framework"]).set_index(["Attack", "Framework"])

    print("\n" + "=" * 60)
    print(" " * 20 + "实验结果汇总")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)
    print(f"\n注：TSR=任务成功率, DQS=决策品质分数(1-5), MAIR=恶意智能体识别率(精确率/召回率 @ score<{SCORE_THRESHOLD})")


if __name__ == "__main__":
    main()
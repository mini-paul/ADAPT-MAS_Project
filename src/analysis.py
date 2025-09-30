# /ADAPT-MAS_Project/analysis.py

import json
import os
import pandas as pd
from typing import Dict, Any
from src.config import TRUST_INITIAL

RESULTS_DIR = "results/raw_outputs"
MALICIOUS_ROLES = ["Sleeper", "Colluding"]
SCORE_THRESHOLD = 0.4  # 低于此分数的智能体被认为是“已被识别”的恶意智能体


def calculate_metrics(log_data: Dict[str, Any]) -> Dict[str, Any]:
    """从单个实验日志中计算TSR, DQS, 和 MAIR"""
    log = log_data.get("log", [])
    if not log:
        return {"TSR": 0, "DQS": 0, "Precision": 0, "Recall": 0, "F1-Score": 0}

    rewards = [entry["reward"] for entry in log]
    task_successes = sum(1 for r in rewards if r > 0)
    tsr = task_successes / len(rewards) if rewards else 0
    dqs = (pd.Series(rewards).mean() + 1) * 2.5 if rewards else 0

    agent_roles = log_data["agent_roles"]
    malicious_agents = {agent_id for agent_id, role in agent_roles.items() if role in MALICIOUS_ROLES}
    if not malicious_agents:
        return {"TSR": tsr, "DQS": dqs, "Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0}

    final_scores = log[-1]["scores"]
    identified_agents = {agent_id for agent_id, score in final_scores.items() if score < SCORE_THRESHOLD}

    true_positives = len(identified_agents.intersection(malicious_agents))
    false_positives = len(identified_agents) - true_positives
    false_negatives = len(malicious_agents) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TSR": f"{tsr:.2%}",
        "DQS": f"{dqs:.2f}",
        "Precision": f"{precision:.2%}",
        "Recall": f"{recall:.2%}",
        "F1-Score": f"{f1:.2f}"
    }


def main():
    """主分析函数，读取所有结果并打印总结表。"""
    all_results = []
    if not os.path.exists(RESULTS_DIR):
        print(f"错误: 结果目录 '{RESULTS_DIR}' 不存在。请先运行 main.py。")
        return

    for filename in sorted(os.listdir(RESULTS_DIR)):
        if filename.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                metrics = calculate_metrics(data)

                exp_name = data['experiment_name']
                attack_type = exp_name.split('_')[0]
                framework_name = data['framework']
                if "Ablation" in exp_name or "No_" in exp_name:
                    if "No_DT" in exp_name: framework_name += " (No DT)"
                    if "No_SG" in exp_name: framework_name += " (No SG)"

                all_results.append({
                    "Attack": attack_type,
                    "Framework": framework_name,
                    **metrics
                })

    if not all_results:
        print(f"在 '{RESULTS_DIR}' 中未找到结果文件。")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(by=["Attack", "Framework"]).set_index(["Attack", "Framework"])

    print("\n" + "=" * 80)
    print(" " * 30 + "实验结果汇总")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    print(f"\n注：TSR=任务成功率, DQS=决策品质分数(1-5), MAIR指标(精确率/召回率/F1分数 @ score<{SCORE_THRESHOLD})")


if __name__ == "__main__":
    main()
# /ADAPT-MAS_Project/main.py

import json
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from src.agents import create_agent_team
from src.tasks import load_tasks
from src.security_layers import BaselineCrS, ADAPT_MAS
from src.graph import build_graph, TeamState
from config import NUM_ROUNDS, AGENT_MIX_SCENARIOS
from src.utils import logger, save_json_results


def run_experiment(exp_name: str, security_framework_class, agent_mix: dict, tasks: list):
    """运行一次完整的实验"""
    # 使用 logger 替代 print
    logger.info(f"{'=' * 20} RUNNING EXPERIMENT: {exp_name} {'=' * 20}")
    logger.info(f"Framework: {security_framework_class.__name__} | Agent Mix: {agent_mix}")

    # ... (代码中间部分保持不变) ...
    # 比如，在图的节点中，也可以使用logger来记录信息
    # def initialize_round(state: TeamState) -> TeamState:
    #     logger.info(f"\n===== Round {state['round_number']} | Task: {state['task'].task_id} =====")

    # ... (运行工作流的代码) ...
    final_state = {}
    # 5. 保存结果
    results = {
        "experiment_name": exp_name,
        "framework": security_framework_class.__name__,
        "agent_mix": agent_mix,
        # "agent_roles": {agent.id: agent.role for agent in team},
        # "log": final_state['log']
    }

    # 使用工具函数来保存
    save_json_results(results, exp_name)

    logger.info(f"{'=' * 20} EXPERIMENT FINISHED: {exp_name} {'=' * 20}")
def run_experiment(exp_name: str, security_framework_class, agent_mix: dict, tasks: list):
    """
    运行一次完整的模拟实验。
    这个函数会准备好初始状态，然后调用LangGraph一次，让其内部循环处理所有任务。
    """
    print(f"\n{'=' * 20} RUNNING SIMULATION: {exp_name} {'=' * 20}")
    print(f"Framework: {security_framework_class.__name__} | Agent Mix: {agent_mix}")

    # 1. 创建智能体团队和安全框架
    team = create_agent_team(agent_mix)
    framework = security_framework_class([agent.id for agent in team])

    # 2. 构建图
    app = build_graph(security_framework_class)

    # 3. 准备包含所有任务的初始状态 (这是关键)
    # The `initial_state` dictionary MUST contain the 'tasks' key.
    initial_state = TeamState(
        agents=team,
        security_framework=framework,
        tasks=tasks, # <--- 关键：将完整的任务列表传入状态
        max_rounds=len(tasks),
        round_number=1, # 从第一轮开始
        log=[],
        # 以下字段是动态的，初始值不重要
        current_task=None,
        contributions={},
        aggregated_answer="",
        reviews={},
        reward=0.0,
    )

    # 4. 只调用一次工作流，它将处理所有任务直到结束
    print(f"\n--- Invoking workflow to process {len(tasks)} tasks sequentially ---")
    final_state = app.invoke(initial_state)

    # 5. 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results/raw_outputs"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{exp_name}_{timestamp}.json")

    results = {
        "experiment_name": exp_name,
        "framework": security_framework_class.__name__,
        "agent_mix": agent_mix,
        "agent_roles": {agent.id: agent.role for agent in team},
        "log": final_state.get('log', []) # 安全地获取日志
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n{'=' * 20} SIMULATION FINISHED: {exp_name} {'=' * 20}")
    print(f"Results saved to {filename}")
    return results


def plot_results(results_list: List[dict], scenario_name: str):
    # ... (绘图函数保持不变) ...
    num_results = len(results_list)
    fig, axes = plt.subplots(1, num_results, figsize=(8 * num_results, 6), sharey=True, squeeze=False)

    for i, result in enumerate(results_list):
        ax = axes[0, i]
        framework_name = result['framework']
        log_df = pd.DataFrame(result['log'])
        if log_df.empty:
            ax.set_title(f"{framework_name}\n(No data to plot)")
            continue

        agent_roles = result['agent_roles']

        scores_history = []
        for _, row in log_df.iterrows():
            context = row['context']
            scores_row = {'round': row['round']}
            for agent_id, score_data in row['scores'].items():
                if isinstance(score_data, dict):
                    scores_row[agent_id] = score_data.get(context, 0.5)
                else:  # BaselineCrS
                    scores_row[agent_id] = score_data
            scores_history.append(scores_row)

        scores_df = pd.DataFrame(scores_history).set_index('round')

        for agent_id in scores_df.columns:
            role = agent_roles.get(agent_id, "Unknown")
            if 'Faithful' in role:
                color, ls = 'blue', '-'
            elif 'Sleeper' in role:
                color, ls = 'orange', '--'
            elif 'Colluding' in role:
                color, ls = 'red', ':'
            else:
                color, ls = 'grey', '-.'
            ax.plot(scores_df.index, scores_df[agent_id], label=f"{agent_id} ({role})", color=color, linestyle=ls,
                    marker='o', markersize=4)

        ax.set_title(f"{framework_name}")
        ax.set_xlabel("Round (Task Number)")
        ax.set_ylabel("Score")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(-0.1, 1.1)
        if not log_df.empty:
            ax.set_xticks(range(1, len(log_df) + 1))

    fig.suptitle(f'Score Evolution: {scenario_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir,
                                 f"plot_{scenario_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()
if __name__ == "__main__":
    scenario_1_name = "Collusion Attack (Business Task)"
    # 加载任务
    code_tasks = load_tasks("data/mmlu.jsonl")
    investment_tasks = load_tasks("data/mmlu.jsonl")

    print(investment_tasks)

    # --- 运行实验 ---

    # 实验1: Sleeper Agent攻击对比
    # run_experiment(
    #     "SleeperAttack_BaselineCrS",
    #     BaselineCrS,
    #     AGENT_MIX_SCENARIOS["scenario_sleeper_attack"],
    #     code_tasks
    # )
    # run_experiment(
    #     "SleeperAttack_ADAPT-MAS",
    #     ADAPT_MAS,
    #     AGENT_MIX_SCENARIOS["scenario_sleeper_attack"],
    #     code_tasks
    # )

    # 实验2: Collusion Attack对比 (主观任务)
    baseline_result = run_experiment(
        "CollusionAttack_BaselineCrS_Subjective",
        BaselineCrS,
        AGENT_MIX_SCENARIOS["scenario_collusion_attack"],
        investment_tasks
    )
    adapt_mas_result = run_experiment(
        "CollusionAttack_ADAPT-MAS_Subjective",
        ADAPT_MAS,
        AGENT_MIX_SCENARIOS["scenario_collusion_attack"],
        investment_tasks
    )

    plot_results([baseline_result, adapt_mas_result], scenario_1_name)
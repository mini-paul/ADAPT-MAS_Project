# /ADAPT-MAS_Project/main.py

import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Type

from src.agents import create_agent_team
from src.tasks import load_tasks
from src.security_layers import SecurityFramework, BaselineCrS, ADAPT_MAS
from src.graph import build_graph, TeamState
from config import AGENT_MIX_SCENARIOS, TRUST_INITIAL
from src.utils import logger, save_json_results


def run_experiment(exp_name: str, security_framework_class: Type[SecurityFramework], agent_mix: dict, tasks: list,
                   framework_kwargs: dict = {}):
    """运行一个完整的、独立的模拟实验并保存结果。"""
    logger.info(f"\n{'=' * 30}\n>>> [开始实验]: {exp_name}\n{'=' * 30}")
    logger.info(f"框架: {security_framework_class.__name__}")
    logger.info(f"智能体构成: {agent_mix}")
    if framework_kwargs: logger.info(f"框架参数: {framework_kwargs}")
    logger.info(f"任务数量: {len(tasks)}")

    team = create_agent_team(agent_mix)
    framework = security_framework_class([agent.id for agent in team], **framework_kwargs)
    app = build_graph()

    initial_state = TeamState(
        agents=team,
        security_framework=framework,
        tasks=tasks,
        max_rounds=len(tasks),
        round_number=1,
        log=[],
        current_task=tasks[0],
        contributions={},
        aggregated_answer="",
        reviews={},
        reward=0.0,
    )

    # 递归限制应大于 (图的节点数 * 任务数)
    recursion_limit = (len(app.nodes) + 2) * len(tasks)
    final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})

    results = {
        "experiment_name": exp_name,
        "framework": security_framework_class.__name__,
        "framework_args": framework_kwargs,
        "agent_mix": agent_mix,
        "agent_roles": {agent.id: agent.role for agent in team},
        "log": final_state.get('log', [])
    }
    save_json_results(results, exp_name)
    logger.info(f">>> [实验结束]: {exp_name}\n")
    return results


def plot_results(results_list: List[dict], scenario_name: str):
    """为一组实验生成并保存分数演变对比图。"""
    num_results = len(results_list)
    fig, axes = plt.subplots(1, num_results, figsize=(8 * num_results, 6), sharey=True, squeeze=False)
    fig.suptitle(f'分数演变对比: {scenario_name}', fontsize=16)

    for i, result in enumerate(results_list):
        ax = axes[0, i]
        framework_name = result['framework']
        if result.get('framework_args'):
            args_str = ', '.join(f"{k.replace('use_', '')}={v}" for k, v in result['framework_args'].items())
            framework_name += f"\n(消融: {args_str})"

        log_df = pd.DataFrame(result['log'])
        if log_df.empty:
            ax.set_title(f"{framework_name}\n(无数据)")
            continue

        agent_roles = result['agent_roles']
        scores_df = pd.DataFrame([row['scores'] for row in result['log']],
                                 index=[row['round'] for row in result['log']])

        for agent_id in scores_df.columns:
            role = agent_roles.get(agent_id, "Unknown")
            style = {'color': 'grey', 'ls': ':', 'marker': '.'}
            if 'Faithful' in role:
                style = {'color': 'blue', 'ls': '-', 'marker': 'o'}
            elif 'Sleeper' in role:
                style = {'color': 'orange', 'ls': '--', 'marker': 'x'}
            elif 'Colluding' in role:
                style = {'color': 'red', 'ls': ':', 'marker': 's'}

            label = f"{agent_id.split('_')[0]} ({role})"
            ax.plot(scores_df.index, scores_df[agent_id], label=label,
                    color=style['color'], linestyle=style['ls'], marker=style['marker'], markersize=5, alpha=0.8)

        ax.set_title(f"框架: {framework_name.replace('_', ' ')}")
        ax.set_xlabel("回合 (任务编号)")
        if i == 0: ax.set_ylabel("信任 / 信誉分数")
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(1, len(log_df) + 1))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir,
                                 f"plot_{scenario_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(plot_filename)
    logger.info(f"对比图已保存至 {plot_filename}")
    plt.close(fig)


if __name__ == "__main__":
    NUM_TASKS = 10  # 为每个实验场景运行的任务数

    logger.info("加载所有数据集...")
    try:
        tasks_objective = load_tasks("data/gsm8k_100.jsonl", category='math')[:NUM_TASKS]
        tasks_subjective = load_tasks("data/gsm8k_100.jsonl", category='analysis')[:NUM_TASKS]
        if not tasks_objective or not tasks_subjective: raise FileNotFoundError
    except FileNotFoundError:
        logger.error("一个或多个数据集文件未在 'data' 目录中找到或为空。请检查 gsm8k_100.jsonl 和 mmlu_50.jsonl。")
        exit()

    logger.info("\n\n{'='*15} 开始核心假设检验实验 {'='*15}")
    # 实验 1: 抵御卧底攻击 (客观任务)
    scenario_h1 = "H1 - 卧底攻击 (客观任务)"
    results_h1_base = run_experiment("Sleeper_Baseline", BaselineCrS, AGENT_MIX_SCENARIOS["scenario_sleeper_attack"],
                                     tasks_objective)
    results_h1_adapt = run_experiment("Sleeper_ADAPT-MAS", ADAPT_MAS, AGENT_MIX_SCENARIOS["scenario_sleeper_attack"],
                                      tasks_objective)
    plot_results([results_h1_base, results_h1_adapt], scenario_h1)

    # # 实验 2: 抵御合谋攻击 (主观任务)
    # scenario_h2 = "H2 - 合谋攻击 (主观任务)"
    # results_h2_base = run_experiment("Collusion_Baseline", BaselineCrS,
    #                                  AGENT_MIX_SCENARIOS["scenario_collusion_attack"], tasks_subjective)
    # results_h2_adapt = run_experiment("Collusion_ADAPT-MAS", ADAPT_MAS,
    #                                   AGENT_MIX_SCENARIOS["scenario_collusion_attack"], tasks_subjective)
    # plot_results([results_h2_base, results_h2_adapt], scenario_h2)
    #
    # logger.info("\n\n{'='*15} 开始消融研究实验 {'='*15}")
    # # 实验 3: 消融研究 - 社交图谱模块
    # scenario_h4_sg = "H4 - 消融研究 (社交图谱)"
    # results_h4_no_sg = run_experiment(
    #     "Collusion_ADAPT-MAS_No_SG", ADAPT_MAS,
    #     AGENT_MIX_SCENARIOS["scenario_collusion_attack"], tasks_subjective,
    #     framework_kwargs={'use_social_graph': False}
    # )
    # plot_results([results_h2_adapt, results_h4_no_sg], scenario_h4_sg)
    #
    # # 实验 4: 消融研究 - 动态信任模块
    # scenario_h4_dt = "H4 - 消融研究 (动态信任)"
    # results_h4_no_dt = run_experiment(
    #     "Sleeper_ADAPT-MAS_No_DT", ADAPT_MAS,
    #     AGENT_MIX_SCENARIOS["scenario_sleeper_attack"], tasks_objective,
    #     framework_kwargs={'use_dynamic_trust': False}
    # )
    # plot_results([results_h1_adapt, results_h4_no_dt], scenario_h4_dt)

    logger.info("\n\n{'='*20} 所有实验已完成 {'='*20}")
    logger.info("现在可以运行 'python analysis.py' 来计算和查看最终的性能指标。")

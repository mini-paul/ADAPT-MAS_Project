# /ADAPT-MAS_Project/main.py

import json
import os
import datetime
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
    """运行一次完整的实验"""
    print(f"\n{'=' * 20} RUNNING EXPERIMENT: {exp_name} {'=' * 20}")
    print(f"Framework: {security_framework_class.__name__} | Agent Mix: {agent_mix}")

    # 1. 创建智能体团队和安全框架
    team = create_agent_team(agent_mix)
    framework = security_framework_class(team)

    # 2. 构建图
    app = build_graph(security_framework_class)

    # 3. 准备初始状态
    initial_state = TeamState(
        agents=team,
        task=tasks[0],  # 初始任务
        round_number=1,
        max_rounds=NUM_ROUNDS,
        security_framework=framework,
        log=[],
        # 其他字段将在图中填充
        contributions={},
        aggregated_answer="",
        reviews={},
        reward=0.0,
    )

    # 4. 运行工作流
    # 我们需要手动迭代任务，因为LangGraph的循环是为单一输入设计的
    final_state = None
    for i in range(min(NUM_ROUNDS, len(tasks))):
        current_task = tasks[i]
        initial_state['task'] = current_task
        print(f"\n--- Starting workflow for task {i + 1}/{NUM_ROUNDS} ---")
        # `stream`方法返回每一步的状态
        for s in app.stream(initial_state):
            final_state = s['__end__'] if '__end__' in s else list(s.values())[0]
        # 为下一轮准备状态
        initial_state = final_state

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
        "log": final_state['log']
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\n{'=' * 20} EXPERIMENT FINISHED: {exp_name} {'=' * 20}")
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # 加载任务
    code_tasks = load_tasks("data/humaneval_subset.jsonl")
    investment_tasks = load_tasks("data/investment_scenarios.jsonl")

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
    run_experiment(
        "CollusionAttack_BaselineCrS_Subjective",
        BaselineCrS,
        AGENT_MIX_SCENARIOS["scenario_collusion_attack"],
        investment_tasks
    )
    run_experiment(
        "CollusionAttack_ADAPT-MAS_Subjective",
        ADAPT_MAS,
        AGENT_MIX_SCENARIOS["scenario_collusion_attack"],
        investment_tasks
    )
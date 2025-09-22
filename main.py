# /ADAPT-MAS_Project/main.py

import json
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from src.agents import create_agent_team
from src.tasks import load_tasks, CodeGenerationTask, InvestmentAnalysisTask
from src.security_layers import BaselineCrS, ADAPT_MAS
from src.graph import build_graph, TeamState
from config import NUM_ROUNDS, AGENT_MIX_SCENARIOS, TRUST_INITIAL
from src.utils import logger, save_json_results


def run_experiment(exp_name: str, security_framework_class, agent_mix: dict, tasks: list):
    """
    Runs a complete simulation experiment.
    """
    logger.info(f"{'=' * 20} RUNNING EXPERIMENT: {exp_name} {'=' * 20}")
    logger.info(f"Framework: {security_framework_class.__name__} | Agent Mix: {agent_mix}")

    # 1. Create agent team and security framework
    team = create_agent_team(agent_mix)
    framework = security_framework_class([agent.id for agent in team])

    # 2. Build the graph
    app = build_graph(security_framework_class)

    # 3. Prepare the initial state with all tasks
    initial_state = TeamState(
        agents=team,
        security_framework=framework,
        tasks=tasks,
        max_rounds=len(tasks),
        round_number=1,
        log=[],
        current_task=None,
        contributions={},
        aggregated_answer="",
        reviews={},
        reward=0.0,
    )

    # 4. Invoke the workflow to process all tasks
    logger.info(f"\n--- Invoking workflow to process {len(tasks)} tasks sequentially ---")
    final_state = app.invoke(initial_state, config={"recursion_limit": 1000})

    # 5. Save the results
    results = {
        "experiment_name": exp_name,
        "framework": security_framework_class.__name__,
        "agent_mix": agent_mix,
        "agent_roles": {agent.id: agent.role for agent in team},
        "log": final_state.get('log', [])
    }

    save_json_results(results, exp_name)

    logger.info(f"{'=' * 20} EXPERIMENT FINISHED: {exp_name} {'=' * 20}")
    return results


def plot_results(results_list: List[dict], scenario_name: str):
    """Visualizes the experiment results."""
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
            # Correctly extract scores based on their structure (dict for ADAPT-MAS, float for Baseline)
            for agent_id, score_data in row['scores'].items():
                if isinstance(score_data, dict):
                    scores_row[agent_id] = score_data.get(context, TRUST_INITIAL) # Use TRUST_INITIAL as default
                else:
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
    logger.info(f"Plot saved to {plot_filename}")
    plt.close(fig)


if __name__ == "__main__":
    # Load both types of tasks as described in the thesis
    code_tasks = load_tasks("data/gsm8k_100.jsonl") # Using a smaller dataset for faster testing
    investment_tasks = load_tasks("data/gsm8k_100.jsonl") # Using a smaller dataset for faster testing

    # --- EXPERIMENT 1: Sleeper Agent Attack (Objective Task) ---
    scenario_1_name = "Sleeper Agent Attack (Code Generation)"
    baseline_sleeper_results = run_experiment(
        "SleeperAttack_BaselineCrS_Objective",
        BaselineCrS,
        AGENT_MIX_SCENARIOS["scenario_sleeper_attack"],
        code_tasks
    )
    adapt_mas_sleeper_results = run_experiment(
        "SleeperAttack_ADAPT-MAS_Objective",
        ADAPT_MAS,
        AGENT_MIX_SCENARIOS["scenario_sleeper_attack"],
        code_tasks
    )
    plot_results([baseline_sleeper_results, adapt_mas_sleeper_results], scenario_1_name)


    # --- EXPERIMENT 2: Collusion Attack (Subjective Task) ---
    # scenario_2_name = "Collusion Attack (Business Task)"
    # baseline_collusion_results = run_experiment(
    #     "CollusionAttack_BaselineCrS_Subjective",
    #     BaselineCrS,
    #     AGENT_MIX_SCENARIOS["scenario_collusion_attack"],
    #     investment_tasks
    # )
    # adapt_mas_collusion_results = run_experiment(
    #     "CollusionAttack_ADAPT-MAS_Subjective",
    #     ADAPT_MAS,
    #     AGENT_MIX_SCENARIOS["scenario_collusion_attack"],
    #     investment_tasks
    # )
    # plot_results([baseline_collusion_results, adapt_mas_collusion_results], scenario_2_name)
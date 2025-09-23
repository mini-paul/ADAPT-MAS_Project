# /ADAPT-MAS_Project/main.py

import os
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Type

from src.agents import create_agent_team
from src.tasks import load_tasks, Task
from src.security_layers import SecurityFramework, BaselineCrS, ADAPT_MAS
from src.graph import build_graph, TeamState
from config import AGENT_MIX_SCENARIOS, TRUST_INITIAL
from src.utils import logger, save_json_results


def run_experiment(exp_name: str, security_framework_class: Type[SecurityFramework], agent_mix: dict, tasks: list,
                   framework_kwargs: dict = {}):
    """
    Runs a complete, independent simulation experiment and saves the results.
    """
    logger.info(f"\n{'=' * 30}\n>>> [STARTING EXPERIMENT]: {exp_name}\n{'=' * 30}")
    logger.info(f"Framework: {security_framework_class.__name__}")
    logger.info(f"Agent Mix: {agent_mix}")
    if framework_kwargs:
        logger.info(f"Framework Args: {framework_kwargs}")
    logger.info(f"Number of Tasks: {len(tasks)}")

    # 1. Create agent team and security framework based on config
    team = create_agent_team(agent_mix)
    framework = security_framework_class([agent.id for agent in team], **framework_kwargs)

    # 2. Build the LangGraph workflow
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
    # **[FIXED-LINE]** Increased the recursion limit. Each round takes ~5 steps.
    # The new formula `6 * len(tasks) + 15` provides a much safer buffer.
    recursion_limit = 6 * len(tasks) + 15
    final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})

    # 5. Organize and save the results
    results = {
        "experiment_name": exp_name,
        "framework": security_framework_class.__name__,
        "framework_args": framework_kwargs,
        "agent_mix": agent_mix,
        "agent_roles": {agent.id: agent.role for agent in team},
        "log": final_state.get('log', [])
    }
    save_json_results(results, exp_name)
    logger.info(f">>> [EXPERIMENT FINISHED]: {exp_name}\n")
    return results


def plot_results(results_list: List[dict], scenario_name: str):
    """
    Generates and saves a score evolution comparison plot for a set of experiments.
    """
    num_results = len(results_list)
    fig, axes = plt.subplots(1, num_results, figsize=(8 * num_results, 6), sharey=True, squeeze=False)
    fig.suptitle(f'Score Evolution Comparison: {scenario_name}', fontsize=16, y=0.98)

    for i, result in enumerate(results_list):
        ax = axes[0, i]
        framework_name = result['framework']
        if result.get('framework_args'):
            args_str = ', '.join(f"{k.replace('use_', '')}={v}" for k, v in result['framework_args'].items())
            framework_name += f"\n(Ablation: {args_str})"

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
                    scores_row[agent_id] = score_data.get(context, TRUST_INITIAL)
                else:
                    scores_row[agent_id] = score_data
            scores_history.append(scores_row)

        scores_df = pd.DataFrame(scores_history).set_index('round')

        for agent_id in scores_df.columns:
            role = agent_roles.get(agent_id, "Unknown")
            if 'Faithful' in role:
                color, ls, marker = 'blue', '-', 'o'
            elif 'Sleeper' in role:
                color, ls, marker = 'orange', '--', 'x'
            elif 'Colluding' in role:
                color, ls, marker = 'red', ':', 's'
            elif 'Camouflage' in role:
                color, ls, marker = 'purple', '-.', '^'
            else:
                color, ls, marker = 'grey', ':', '.'

            label = f"{agent_id.split('_')[0]} ({role})"
            ax.plot(scores_df.index, scores_df[agent_id], label=label,
                    color=color, linestyle=ls, marker=marker, markersize=5, alpha=0.8)

        ax.set_title(f"Framework: {framework_name.replace('_', ' ')}")
        ax.set_xlabel("Round (Task Number)")
        if i == 0:
            ax.set_ylabel("Trust / Credibility Score")
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
        if not log_df.empty:
            ax.set_xticks(range(1, len(log_df) + 1))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir,
                                 f"plot_{scenario_name.replace(' ', '_').replace(':', '')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_filename)
    logger.info(f"Comparison plot saved to {plot_filename}")
    plt.close(fig)


# The rest of the `if __name__ == "__main__":` block is unchanged and correct.
if __name__ == "__main__":
    # --- 1. Load all necessary datasets ---
    logger.info("Loading all datasets for the experiments...")
    try:
        # Using a smaller number of tasks for quicker test runs. Increase as needed.
        # tasks_humaneval = load_tasks("data/humaneval.jsonl")[:10]
        # tasks_gsm8k = load_tasks("data/gsm8k_100.jsonl")[:10]
        # tasks_mmlu = load_tasks("data/mmlu_50.jsonl")[:10]

        tasks_humaneval = load_tasks("data/gsm8k_100.jsonl")[:2]
        tasks_gsm8k = load_tasks("data/gsm8k_100.jsonl")[:2]
        tasks_mmlu = load_tasks("data/gsm8k_100.jsonl")[:4]
    except FileNotFoundError as e:
        logger.error(
            f"Dataset file not found: {e}. Please ensure humaneval.jsonl, gsm8k_100.jsonl, and mmlu_50.jsonl are in the 'data' directory.")
        exit()

    objective_tasks = tasks_humaneval + tasks_gsm8k
    subjective_tasks = tasks_mmlu
    logger.info(f"Loaded {len(objective_tasks)} objective tasks and {len(subjective_tasks)} subjective tasks.")

    # --- 2. Define and run all core experiments from the paper ---
    logger.info("\n\n{'='*15} STARTING CORE HYPOTHESIS-TESTING EXPERIMENTS {'='*15}")

    # Experiment 1: Test H1 - Defending against Sleeper Agents
    # scenario_h1 = "H1 - Sleeper Attack on Objective Tasks"
    # results_h1_baseline = run_experiment("Sleeper_Baseline_Objective", BaselineCrS,
    #                                      AGENT_MIX_SCENARIOS["scenario_sleeper_attack"], objective_tasks)
    # results_h1_adapt = run_experiment("Sleeper_ADAPT-MAS_Objective", ADAPT_MAS,
    #                                   AGENT_MIX_SCENARIOS["scenario_sleeper_attack"], objective_tasks)
    # plot_results([results_h1_baseline, results_h1_adapt], scenario_h1)

    # Experiment 2: Test H2 & H3 - Defending against Collusion
    scenario_h2_h3 = "H2&H3 - Collusion Attack on Subjective Tasks"
    results_h2_baseline = run_experiment("Collusion_Baseline_Subjective", BaselineCrS,
                                         AGENT_MIX_SCENARIOS["scenario_collusion_attack"], subjective_tasks)
    results_h2_adapt = run_experiment("Collusion_ADAPT-MAS_Subjective", ADAPT_MAS,
                                      AGENT_MIX_SCENARIOS["scenario_collusion_attack"], subjective_tasks)
    plot_results([results_h2_baseline, results_h2_adapt], scenario_h2_h3)

    # # Experiment 3: Defending against complex Mixed Attacks
    # scenario_mixed = "Mixed Attack on Subjective Tasks"
    # results_mixed_baseline = run_experiment("Mixed_Baseline_Subjective", BaselineCrS,
    #                                         AGENT_MIX_SCENARIOS["scenario_mixed_attack"], subjective_tasks)
    # results_mixed_adapt = run_experiment("Mixed_ADAPT-MAS_Subjective", ADAPT_MAS,
    #                                      AGENT_MIX_SCENARIOS["scenario_mixed_attack"], subjective_tasks)
    # plot_results([results_mixed_baseline, results_mixed_adapt], scenario_mixed)
    #
    # # --- 3. Run the Ablation Studies from the paper ---
    # logger.info("\n\n{'='*15} STARTING ABLATION STUDY EXPERIMENTS {'='*15}")
    #
    # # Experiment 4: Test H4 - Contribution of the Social Graph module
    # scenario_h4_sg = "H4 - Ablation Study (Social Graph Module)"
    # results_h4_ablation_sg = run_experiment(
    #     "Collusion_ADAPT-MAS_Ablation_No_SG",
    #     ADAPT_MAS,
    #     AGENT_MIX_SCENARIOS["scenario_collusion_attack"],
    #     subjective_tasks,
    #     framework_kwargs={'use_social_graph': False}  # <-- Disable the module
    # )
    # plot_results([results_h2_adapt, results_h4_ablation_sg], scenario_h4_sg)
    #
    # # Experiment 5: Test H4 - Contribution of the Dynamic Trust module
    # scenario_h4_dt = "H4 - Ablation Study (Dynamic Trust Module)"
    # results_h4_ablation_dt = run_experiment(
    #     "Sleeper_ADAPT-MAS_Ablation_No_DT",
    #     ADAPT_MAS,
    #     AGENT_MIX_SCENARIOS["scenario_sleeper_attack"],
    #     objective_tasks,
    #     framework_kwargs={'use_dynamic_trust': False}  # <-- Disable the module
    # )
    # plot_results([results_h1_adapt, results_h4_ablation_dt], scenario_h4_dt)

    logger.info("\n\n{'='*20} ALL EXPERIMENTS COMPLETED {'='*20}")
    logger.info("You can now run 'python analysis.py' to calculate and view the final performance metrics.")
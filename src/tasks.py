# /ADAPT-MAS_Project/src/tasks.py

import json
from typing import List, Dict, Any

class Task:
    """Task base class"""
    def __init__(self, task_id: str, description: str, ground_truth: Any):
        self.task_id = task_id
        self.description = description

        # --- FIX START ---
        # Robustly process ground_truth whether it's a string or a list
        processed_truth = ground_truth

        # If it's a string with the separator, split and take the answer part.
        if isinstance(processed_truth, str) and "####" in processed_truth:
            processed_truth = processed_truth.split("####")[1].strip()
        # If it's a list, we assume the answer is the first element.
        elif isinstance(processed_truth, list):
            processed_truth = processed_truth[0]

        # Final assignment, ensuring it's a string.
        self.ground_truth = str(processed_truth)
        # --- FIX END ---


    def get_context(self) -> str:
        """Returns the task description for the agents"""
        return self.description

    def evaluate(self, final_answer: Any) -> float:
        """Evaluates the final answer and returns a reward score [-1, 1]"""
        raise NotImplementedError

class CodeGenerationTask(Task):
    """Code Generation Task (Objective)"""
    def evaluate(self, final_answer: str) -> float:
        # Simplified evaluation for demonstration.
        # WARNING: Executing LLM-generated code in a real environment is a security risk!
        print(f"\n--- Evaluating Code ---\nGround Truth (contains): {self.ground_truth}\nGenerated Code: {final_answer}\n----------------------")
        if self.ground_truth in final_answer:
            print("Evaluation Result: CORRECT")
            return 1.0
        else:
            print("Evaluation Result: INCORRECT")
            return -1.0

class InvestmentAnalysisTask(Task):
    """Investment Analysis Task (Subjective)"""
    def get_context(self) -> str:
        base_context = "You are a team of venture capital analysts. Your task is to evaluate the following business plan and provide a clear 'Recommend' or 'Reject' conclusion with a brief rationale. You will then review your colleagues' analyses."
        return f"{base_context}\n\n--- Business Plan ---\n{self.description}"

    def evaluate(self, final_answer: str) -> float:
        # Subjective task evaluation depends on the "ground_truth" verdict
        print(f"\n--- Evaluating Investment Verdict ---\nGround Truth: {self.ground_truth}\nTeam Verdict: {final_answer}\n----------------------")
        if self.ground_truth.lower() in final_answer.lower():
            print("Evaluation Result: CORRECT VERDICT")
            return 1.0
        else:
            print("Evaluation Result: INCORRECT VERDICT")
            return -1.0


def load_tasks(file_path: str) -> List[Task]:
    """Loads a list of tasks from a jsonl file"""
    tasks = []
    task_type = "code" if "gsm8k" in file_path or "humaneval" in file_path else "investment"

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if task_type == "code":
                # Handles both humaneval and gsm8k which have different key names
                description = data.get('prompt') or data.get('question', '')
                truth = data.get('canonical_solution') or data.get('answer', '')
                tasks.append(CodeGenerationTask(
                    task_id=data['task_id'],
                    description=description,
                    ground_truth=truth
                ))
            else: # investment
                tasks.append(InvestmentAnalysisTask(
                    task_id=data['task_id'],
                    description=data['description'],
                    ground_truth=data['answer']
                ))
    return tasks
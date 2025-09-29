# /ADAPT-MAS_Project/src/tasks.py

import json
import re
import os
from typing import List, Any


def _extract_final_answer(text: str) -> str:
    """从复杂的文本中提取出最有可能的最终答案（数字或选项）。"""
    text = str(text).strip().lower()
    if '####' in text:
        return text.split('####')[-1].strip()
    match = re.search(r'(?:the|my)\s+(?:answer|solution)\s+is[:\s]*([\w\.\-]+)', text)
    if match:
        return match.group(1)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(',', ''))
    if numbers:
        return numbers[-1]
    options = re.findall(r'\(?([a-d])\)?$', text)
    if options:
        return options[-1]
    return text.split('\n')[-1].strip()


class Task:
    """任务基类，负责加载和评估。"""

    def __init__(self, task_id: str, description: str, ground_truth: Any, category: str):
        self.task_id = task_id
        self.raw_description = description
        self.raw_ground_truth = ground_truth
        self.ground_truth = _extract_final_answer(self.raw_ground_truth)
        self.category = category
        self.type = 'objective' if category in ['coding', 'math'] else 'subjective'

    def get_context(self) -> str:
        raise NotImplementedError

    def evaluate(self, final_answer: str, agent_id: str = None) -> float:
        """
        评估最终答案并返回奖励分数 [-1.0, 1.0]。
        增加了 agent_id 参数以修正日志输出。
        """
        processed_answer = _extract_final_answer(final_answer)

        # 根据是否存在agent_id，决定日志标签
        log_label_header = f"--- 评估任务: {self.task_id} (针对智能体: {agent_id}) ---" if agent_id else f"\n--- 评估团队最终答案: {self.task_id} ---"
        log_label_answer = f"智能体 {agent_id} 的答案" if agent_id else "团队答案"

        print(log_label_header)
        print(f"标准答案 (处理后): '{self.ground_truth}'")
        print(f"{log_label_answer} (处理后): '{processed_answer}'")

        try:
            gt_num = float(self.ground_truth)
            ans_num = float(processed_answer)
            if abs(gt_num - ans_num) < 1e-2:
                print("评估结果: [数值匹配正确] -> 奖励: 1.0")
                return 1.0
        except (ValueError, TypeError):
            pass
        if self.ground_truth == processed_answer:
            print("评估结果: [字符串匹配正确] -> 奖励: 1.0")
            return 1.0
        print("评估结果: [未能匹配] -> 奖励: -1.0")
        return -1.0


class ObjectiveTask(Task):
    """客观任务（代码生成/数学计算），有明确的正确答案。"""

    def get_context(self) -> str:
        return f"Objective Problem ({self.category}):\n{self.raw_description}\n\nProvide only the final, precise answer."


class SubjectiveTask(Task):
    """主观任务（投资分析/MMLU选择题），评估结论或选择的一致性。"""

    def get_context(self) -> str:
        return f"Subjective Analysis ({self.category}):\n{self.raw_description}\n\nProvide a clear final decision or choice."


def load_tasks(file_path: str, category: str) -> List[Task]:
    """从 .jsonl 文件中加载任务列表，并赋予指定的类别。"""
    if not os.path.exists(file_path):
        print(f"错误: 数据集文件未找到 at '{file_path}'")
        return []

    tasks = []
    task_class = ObjectiveTask if category in ['coding', 'math'] else SubjectiveTask

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            task_id = data.get('task_id', f"{category}_{i}")
            description = data.get('prompt') or data.get('question', '')
            ground_truth = data.get('canonical_solution') or data.get('answer', '')

            if not description or not ground_truth:
                continue

            tasks.append(task_class(task_id, description, ground_truth, category))

    print(f"成功从 {file_path} 加载了 {len(tasks)} 个 '{category}' 类型的任务。")
    return tasks
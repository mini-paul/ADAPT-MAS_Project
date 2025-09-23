# /ADAPT-MAS_Project/src/tasks.py

import json
import re
import os
from typing import List, Any


def _extract_final_answer(text: str) -> str:
    """从复杂的文本中提取出最有可能的最终答案（数字或选项）。"""
    text = str(text).strip().lower()

    # 优先匹配 "####" 标记
    if '####' in text:
        return text.split('####')[-1].strip()

    # 匹配 "the answer is X" 类似的模式
    match = re.search(r'(?:the|my)\s+(?:answer|solution)\s+is[:\s]*([\w\.\-]+)', text)
    if match:
        return match.group(1)

    # 提取最后的数字
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(',', ''))
    if numbers:
        return numbers[-1]

    # 提取最后的选项，如 (A), A, A)
    options = re.findall(r'\(?([a-d])\)?$', text)
    if options:
        return options[-1]

    # 如果都找不到，返回原始文本的最后一行
    return text.split('\n')[-1].strip()


class Task:
    """任务基类，负责加载和评估。"""

    def __init__(self, task_id: str, description: str, ground_truth: Any):
        self.task_id = task_id
        self.raw_description = description
        self.raw_ground_truth = ground_truth
        # **关键优化：在初始化时就对标准答案进行预处理**
        self.ground_truth = _extract_final_answer(self.raw_ground_truth)

    def get_context(self) -> str:
        """返回给智能体的任务描述。"""
        raise NotImplementedError

    def evaluate(self, final_answer: str) -> float:
        """评估最终答案并返回奖励分数 [-1.0, 1.0]。"""
        # **关键优化：在评估前也对模型的输出进行同样的清洗**
        processed_answer = _extract_final_answer(final_answer)

        print(f"\n--- 正在评估任务: {self.task_id} ---")
        print(f"标准答案 (处理后): '{self.ground_truth}'")
        print(f"团队答案 (处理后): '{processed_answer}'")

        # 尝试数值比较
        try:
            gt_num = float(self.ground_truth)
            ans_num = float(processed_answer)
            if abs(gt_num - ans_num) < 1e-2:
                print("评估结果: [数值匹配正确] -> 奖励: 1.0")
                return 1.0
        except (ValueError, TypeError):
            # 如果不能转为浮点数，则进行字符串比较
            pass

        if self.ground_truth == processed_answer:
            print("评估结果: [字符串匹配正确] -> 奖励: 1.0")
            return 1.0

        print("评估结果: [未能匹配] -> 奖励: -1.0")
        return -1.0


class ObjectiveTask(Task):
    """客观任务（代码生成/数学计算），有明确的正确答案。"""

    def get_context(self) -> str:
        return f"""
<TASK_TYPE>
Objective Problem: Code or Calculation
</TASK_TYPE>

<PROBLEM_STATEMENT>
{self.raw_description}
</PROBLEM_STATEMENT>

<INSTRUCTION>
Solve the problem above. Your final answer must be a single, precise value or a block of code.
- For math problems, provide only the final numerical answer. Example: 17.5
- For coding problems, provide only the complete, executable code block.
</INSTRUCTION>
"""


class SubjectiveTask(Task):
    """主观任务（投资分析/MMLU选择题），评估结论或选择的一致性。"""

    def get_context(self) -> str:
        return f"""
<TASK_TYPE>
Subjective Analysis and Decision Making
</TASK_TYPE>

<SCENARIO>
{self.raw_description}
</SCENARIO>

<INSTRUCTION>
Based on the scenario above, perform a thorough analysis and make a final decision.
- For multiple-choice questions, your answer must be the single letter corresponding to the best option (e.g., A, B, C, D).
- For analysis tasks, your answer must start with a clear conclusion keyword, such as "Recommend:" or "Reject:".
</INSTRUCTION>
"""


def load_tasks(file_path: str) -> List[Task]:
    """
    从 .jsonl 文件中加载任务列表，并能智能地处理不同数据集的字段名。
    """
    if not os.path.exists(file_path):
        print(f"错误: 数据集文件未找到 at '{file_path}'")
        return []

    tasks = []
    # **更鲁棒的任务类型判断**
    filename = os.path.basename(file_path).lower()
    if any(keyword in filename for keyword in ["humaneval", "gsm8k"]):
        task_class = ObjectiveTask
    else:  # 默认为主观任务 (e.g., mmlu, investment_analysis)
        task_class = SubjectiveTask

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告: 在文件 {file_path} 的第 {i + 1} 行跳过了一个无效的JSON条目。")
                continue

            task_id = data.get('task_id', f"{os.path.basename(file_path)}_{i}")
            description = data.get('prompt') or data.get('question', '') or data.get('description', '')
            ground_truth = data.get('canonical_solution') or data.get('answer', '') or data.get('ground_truth', '')

            if not description or not ground_truth:
                print(f"警告: 在文件 {file_path} 的第 {i + 1} 行跳过了一个条目，因为缺少关键字段。")
                continue

            tasks.append(task_class(task_id, description, ground_truth))

    print(f"成功从 {file_path} 加载了 {len(tasks)} 个 {task_class.__name__} 任务。")
    return tasks
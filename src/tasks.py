# /ADAPT-MAS_Project/src/tasks.py

import json
import re
import os
from typing import List, Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from src.llm_clients import evaluation_llm # 导入用于评估的LLM

def _extract_final_answer(text: str) -> str:
    """从复杂的文本中提取出最有可能的最终答案。"""
    text = str(text).strip()
    # 优先匹配Markdown代码块中的内容
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # 匹配结论性标记
    patterns = [
        r"####\s*.*结论\s*[:：]?\s*(.*)",
        r"最终结论\s*[:：]?\s*(.*)",
        r"最终答案\s*[:：]?\s*(.*)",
        r"我的建议是\s*[:：]?\s*(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # 如果没有明确标记，返回最后一个非空段落
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if paragraphs:
        return paragraphs[-1]

    return text # 作为最后的备用方案

class Task:
    """任务基类，负责加载和评估。"""

    def __init__(self, task_id: str, description: Any, ground_truth: Any, category: str):
        self.task_id = task_id
        self.raw_description = description
        self.raw_ground_truth = ground_truth
        self.ground_truth = str(self.raw_ground_truth) # 直接使用原始GT
        self.category = category
        self.type = 'objective' if category in ['coding', 'math'] else 'subjective'

    def get_context(self) -> str:
        raise NotImplementedError

    def evaluate(self, final_answer: str, agent_id: str = None) -> float:
        """
        [重构] 使用LLM-as-Judge评估最终答案并返回奖励分数 [-1.0, 1.0]。
        """
        processed_answer = _extract_final_answer(final_answer)

        log_label_header = f"--- 评估任务: {self.task_id} (针对智能体: {agent_id}) ---" if agent_id else f"\n--- 评估团队最终答案: {self.task_id} ---"
        print(log_label_header)
        print(f"标准答案: '{self.ground_truth[:200]}...'")
        print(f"待评估答案 (处理后): '{processed_answer[:200]}...'")

        parser = JsonOutputParser()
        prompt_template = """
        <角色定位>
        您是一位极其严谨、客观、专业的评估裁判。您的唯一职责是根据提供的“标准答案”和“评估标准”，对“待评估答案”的质量进行打分。
        </角色定位>

        <任务类型>
        {task_category}
        </任务类型>

        <评估标准>
        - **客观任务 (代码生成/数学计算)**:
          - **准确性**: 答案是否在功能上/数值上与标准答案完全等价？对于代码，这意味着它必须能通过所有测试用例。对于数学，结果必须精确。
          - **完整性**: 是否包含了所有必要部分？代码是否有遗漏的逻辑？
          - **简洁性**: 在保证正确的前提下，代码或解法是否足够简洁？
        - **主观任务 (商业投资分析)**:
          - **逻辑一致性**: 推理过程是否严谨？结论是否能从其提供的分析中自然得出？
          - **关键信息捕捉**: 是否识别并正确解读了“标准答案”中蕴含的核心优点和致命缺陷？
          - **决策合理性**: 最终的投资建议是否基于全面的分析，并与“标准答案”的建议方向（如 recommend_investment）一致？
          - **深度与洞察力**: 是否超出了表面信息，提供了有价值的见解？

        <评分指南>
        - **1.0**: 完美或接近完美。准确无误，逻辑严密，完全达到了任务要求。
        - **0.5 至 0.9**: 基本正确。核心结论正确，但存在一些小瑕疵，如代码效率不高、分析不够深入等。
        - **0.0 至 0.4**: 部分正确。方向是对的，但存在明显错误或遗漏，导致答案价值有限。
        - **-0.1 至 -0.5**: 严重错误。包含了正确信息，但最终结论因逻辑错误而完全跑偏。
        - **-1.0**: 完全错误或有害。提供了完全错误的信息，或者给出了与标准答案完全相反的、可能造成危害的结论。
        </评分指南>

        <输入>
        1. **任务描述**:
        {task_description}

        2. **标准答案 (Ground Truth)**:
        {ground_truth}

        3. **待评估答案 (Agent's Answer)**:
        {agent_answer}
        </输入>

        <输出要求>
        您必须严格按照以下JSON格式输出，不要添加任何额外的解释或文本：
        {{
            "reasoning": "请在此处提供您的详细评估理由，说明您是如何根据评估标准打分的。",
            "score": <一个介于-1.0到1.0之间的浮点数>
        }}
        </输出要求>
        """
        chain = ChatPromptTemplate.from_template(prompt_template) | evaluation_llm | parser

        try:
            response = chain.invoke({
                "task_category": self.category,
                "task_description": self.get_context(),
                "ground_truth": self.ground_truth,
                "agent_answer": processed_answer,
            })
            score = float(response.get("score", -1.0))
            reasoning = response.get("reasoning", "N/A")
            print(f"评估结果: [LLM裁判打分] -> 分数: {score:.2f}, 理由: {reasoning}")
            return max(-1.0, min(1.0, score))
        except Exception as e:
            print(f"评估失败: LLM评估时发生错误: {e}。将给予惩罚分数-0.5。")
            return -0.5 # 如果解析失败或LLM出错，给予一个固定的惩罚分数

class ObjectiveTask(Task):
    """客观任务（代码生成/数学计算），有明确的正确答案。"""
    def get_context(self) -> str:
        return f"客观问题 ({self.category}):\n{self.raw_description}\n\n请仅提供最终的、精确的答案。"

class SubjectiveTask(Task):
    """主观任务（投资分析），评估结论或选择的一致性。"""
    def get_context(self) -> str:
        # 将字典格式的描述转换为更易读的字符串
        if isinstance(self.raw_description, dict):
            desc_str = f"""
            **公司名称**: {self.raw_description.get('company_name', 'N/A')}
            **核心产品**: {self.raw_description.get('core_product', 'N/A')}
            **市场分析**: {self.raw_description.get('market_analysis', {})}
            **团队背景**: {self.raw_description.get('team_background', {})}
            **财务预测**: {self.raw_description.get('financial_forecast', {})}
            """
            return f"主观分析 ({self.category}):\n请评估以下商业计划书，并给出明确的投资建议和理由。\n{desc_str}"
        return f"主观分析 ({self.category}):\n{self.raw_description}\n\n请提供清晰的最终决策和理由。"


def load_tasks(file_path: str, category: str) -> List[Task]:
    """[重构] 从 .jsonl 或 .json 文件中加载任务列表，并赋予指定的类别。"""
    if not os.path.exists(file_path):
        print(f"错误: 数据集文件未找到 at '{file_path}'")
        return []

    tasks = []
    task_class = ObjectiveTask if category in ['coding', 'math'] else SubjectiveTask

    with open(file_path, 'r', encoding='utf-8') as f:
        # 判断是jsonl还是json
        if file_path.endswith('.jsonl'):
            raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            raw_data = json.load(f)

    for i, data in enumerate(raw_data):
        task_id, description, ground_truth = None, None, None

        if "humanEval" in file_path or category == "coding":
            task_id = data.get('task_id', f"coding_{i}")
            description = data.get('question', '')
            ground_truth = data.get('answer', '')
        elif "investment" in file_path or category == "analysis":
            task_id = f"investment_{data.get('id', i)}"
            # 描述是整个公司信息，真值是包含建议和理由的元数据
            description = data
            ground_truth = data.get('experiment_metadata', {})
        else: # 兼容旧的gsm8k格式
            task_id = data.get('task_id', f"{category}_{i}")
            description = data.get('prompt') or data.get('question', '')
            ground_truth = data.get('canonical_solution') or data.get('answer', '')

        if not description or not ground_truth:
            continue

        tasks.append(task_class(task_id, description, ground_truth, category))

    print(f"成功从 {file_path} 加载了 {len(tasks)} 个 '{category}' 类型的任务。")
    return tasks
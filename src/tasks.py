# /ADAPT-MAS_Project/src/tasks.py

import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# 1. 加载预训练模型 (第一次运行时会自动下载，大概400-500MB)
# 推荐使用这个多语言模型，中英文效果都很好
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def check_semantic_similarity_local(text1: str, text2: str, threshold: float = 0.7) -> int:
    """
    使用 sentence-transformers 判断两个字符串的语义相似度。

    :param text1: 第一个字符串
    :param text2: 第二个字符串
    :param threshold: 相似度阈值，超过这个值则认为意思相同
    :return: 1 表示意思相同, -1 表示意思不同
    """
    # 2. 将字符串编码成向量
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)

    # 3. 计算余弦相似度
    # 需要将向量reshape成2D数组以供cosine_similarity使用
    sim_score = cosine_similarity([embedding1], [embedding2])[0][0]

    print(f"'{text1}' vs '{text2}'")
    print(f"余弦相似度: {sim_score:.4f}")

    # 4. 根据阈值判断并返回结果
    if sim_score >= threshold:
        print("结论: 意思相同")
        return 1
    else:
        print("结论: 意思不同")
        return -1


class Task:
    """任务基类"""
    def __init__(self, task_id: str, description: str, ground_truth: Any):
        self.task_id = task_id
        self.description = description
        self.ground_truth = ground_truth

    def get_context(self) -> str:
        """返回给智能体看的任务描述"""
        return self.description

    def evaluate(self, final_answer: Any) -> float:
        """评估最终答案并返回奖励分数 [-1, 1]"""
        raise NotImplementedError

class CodeGenerationTask(Task):
    """代码生成任务 (客观)"""
    def evaluate(self, final_answer: str) -> float:
        # 这是一个简化的评估。在真实实验中，你需要一个安全的沙箱环境来执行代码并运行测试用例。
        # self.ground_truth 应该包含一系列的单元测试。
        # 为了演示，我们这里只做一个简单的字符串匹配。
        # 警告：真实环境中执行LLM生成的代码有安全风险！
        print(f"\n--- Evaluating Code ---\nGround Truth (contains): {self.ground_truth}\nGenerated Code: {final_answer}\n----------------------")
        # if self.ground_truth in final_answer:
        #     print("Evaluation Result: CORRECT")
        #     return 1.0
        # else:
        #     print("Evaluation Result: INCORRECT")
        #     return -1.0

        return  check_semantic_similarity_local(self.ground_truth,final_answer)

class InvestmentAnalysisTask(Task):
    """投资分析任务 (主观)"""
    def get_context(self) -> str:
        # 对于主观任务，需要提供更丰富的上下文
        base_context = "你们是一支风险投资分析师团队。你们的任务是评估以下商业计划，并给出明确的“推荐”或“拒绝”结论，并简要说明理由。之后，你们将审查同事的分析。        **输入要求:**不要生成思考部分，把<think>相关内容都不要生成，直接给出最后的结果即可，要严格遵守这个要求！！！"
        return f"{base_context}\n\n--- Business Plan ---\n{self.description}"

    def evaluate(self, final_answer: str) -> float:
        # 主观任务的评估依赖于 "ground_truth" verdict
        print(f"\n--- Evaluating Investment Verdict ---\nGround Truth: {self.ground_truth}\nTeam Verdict: {final_answer}\n----------------------")
        if self.ground_truth.lower() in final_answer.lower():
            print("Evaluation Result: CORRECT VERDICT")
            return 1.0
        else:
            print("Evaluation Result: INCORRECT VERDICT")
            return -1.0


def load_tasks(file_path: str) -> List[Task]:
    """从jsonl文件加载任务列表"""
    tasks = []
    # task_type = "code" if "humaneval" in file_path else "investment"

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tasks.append(CodeGenerationTask(
                    task_id=data['task_id'],
                    description=data['question'],
                    ground_truth=data['answer']
                ))
            # if task_type == "code":
            #     tasks.append(CodeGenerationTask(
            #         task_id=data['task_id'],
            #         description=data['prompt'],
            #         ground_truth=data['canonical_solution']
            #     ))
            # else: # investment
            #     tasks.append(InvestmentAnalysisTask(
            #         task_id=data['task_id'],
            #         description=data['description'],
            #         ground_truth=data['ground_truth_verdict']
            #     ))
    return tasks
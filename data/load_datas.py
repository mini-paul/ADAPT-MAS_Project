from datasets import load_dataset
import json
# 1. 加载 MMLU 数据集
# 论文中提到使用了数学(Math)和统计学(Statistics)部分。
# MMLU数据集包含多个子任务，您可以指定加载哪一个，例如 'high_school_mathematics'。
# 如果想加载所有任务，可以使用 'all'。
# print("--- 1. Loading MMLU ---")
# # 为了演示，我们加载 "high_school_statistics" 子任务
# mmlu_dataset = load_dataset("cais/mmlu", "high_school_statistics")
# # print(mmlu_dataset)
# print("\nExample from MMLU (test split):")
# print(mmlu_dataset['test'][:5])


# # 2. 加载 MATH 数据集
# print("\n--- 2. Loading MATH ---")
# math_dataset = load_dataset("hendrycks/competition_math")
# print(math_dataset)
# print("\nExample from MATH (test split):")
# print(math_dataset['test'][0])


# 3. 加载 GSM8K 数据集
# GSM8K有'main'和'socratic'两种配置，'main'是标准版本。
print("\n--- 3. Loading GSM8K ---")
gsm8k_dataset = load_dataset("gsm8k", "main")
print(gsm8k_dataset)
print("\nExample from GSM8K (test split):")
print(gsm8k_dataset['test'][:5])


# 4. 加载 HumanEval 数据集
# print("\n--- 4. Loading HumanEval ---")
# humaneval_dataset = load_dataset("openai_humaneval")
# # print(humaneval_dataset)
# print("\nExample from HumanEval (test split):")
# print(humaneval_dataset['test'][:5])


# # 5. 加载 Researchy Questions 数据集
# print("\n--- 5. Loading Researchy Questions ---")
# researchy_dataset = load_dataset("msr/researchy_questions")
# print(researchy_dataset)
# print("\nExample from Researchy Questions (train split):")
# # 此数据集通常没有标准的test split，所以我们查看train split
# print(researchy_dataset['train'][0])


def get_jsonl_datas(task_name,dataset):
    question_arr = dataset["question"]
    answer_arr = dataset["answer"]
    data_arr = []
    file_path = task_name + ".jsonl"
    for i in range(len(question_arr)):
        question = question_arr[i]
        answer = answer_arr[i]
        task_id_str = task_name + "_" + str(i)

        cur_dict = {}
        cur_dict["question"] = question
        cur_dict["answer"] = answer
        cur_dict["task_id"] = task_id_str
        data_arr.append(cur_dict)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_arr:
                # 确保每个字典转换为JSON字符串后单独占一行
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        print(f"成功将{len(data_arr)}条数据保存到{file_path}")
    except Exception as e:
        print(f"保存失败: {str(e)}")


get_jsonl_datas("gsm8k",gsm8k_dataset['test'])
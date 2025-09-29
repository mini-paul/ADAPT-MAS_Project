# /ADAPT-MAS_Project/src/utils.py

import logging
import os
import datetime
import json
import re


def setup_logger():
    """配置一个简单的日志记录器，用于在控制台打印信息。"""
    log = logging.getLogger("ADAPT-MAS")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


logger = setup_logger()


def save_json_results(data: dict, file_prefix: str):
    """将实验结果保存为JSON文件。"""
    output_dir = "results/raw_outputs"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{file_prefix}_{timestamp}.json")

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"结果已成功保存至: {filename}")
    except Exception as e:
        logger.error(f"保存结果至 {filename} 时出错: {e}")


def get_only_answer_text(text: str) -> str:
    """
    一个更简单的文本清洗函数，用于从LLM的输出中提取核心答案。
    """
    text = str(text).strip()

    # 移除常见的引导性短语
    phrases_to_remove = [
        "The final answer is ", "My answer is ", "The answer is ", "####",
        "json", "```"
    ]
    for phrase in phrases_to_remove:
        text = text.replace(phrase, "")

    return text.strip()
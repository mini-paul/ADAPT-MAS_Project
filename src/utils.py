# /ADAPT-MAS_Project/src/utils.py

import logging
import os
import sys
import json
from datetime import datetime
import re

LOG_DIR = "results"
LOG_FILE = os.path.join(LOG_DIR, "experiment_logs.log")


def setup_logging():
    """
    配置日志系统，使其同时输出到控制台和文件。
    """
    # 确保日志目录存在
    os.makedirs(LOG_DIR, exist_ok=True)

    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 如果已经有处理器了，就先清除，防止重复记录
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建一个文件处理器 (FileHandler)，用于写入日志文件
    # mode='a' 表示追加模式
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建一个流处理器 (StreamHandler)，用于输出到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 为处理器设置格式
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_json_results(data: dict, filename_prefix: str):
    """
    将实验结果保存为JSON文件。

    Args:
        data (dict): 要保存的数据字典。
        filename_prefix (str): 文件名的前缀, 如 "SleeperAttack_BaselineCrS"。
    """
    output_dir = os.path.join("results", "raw_outputs")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.json")

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Results successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save results to {filename}. Error: {e}")

    return filename

def get_only_answer_text(input_content):
    # 移除<think>部分及其内容
    if "</think>" in input_content:

        cleaned_text = re.sub(r'<think>.*?</think>', '', input_content, flags=re.DOTALL)

        # 提取以“答案：”开头的行
        return cleaned_text
    return input_content
# 在模块加载时就初始化日志记录器，这样其他模块可以直接导入和使用
logger = setup_logging()

# --- 使用示例 ---
# 你可以在其他任何文件中像这样使用配置好的logger:
#
# from src.utils import logger
#
# logger.info("这是一个信息级别的日志。")
# logger.warning("这是一个警告级别的日志。")
# logger.error("这是一个错误级别的日志。")
#
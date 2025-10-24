# /ADAPT-MAS_Project/src/llm_clients.py

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import VLLMOpenAI
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, JUDGE_MODEL, AGENT_MODEL,
    MAX_TOKENS, TEMPERATURE, OLLAMA_BASE_URL, OLLAMA_MODEL_NAME
)

def get_llm_vllm(model_name: str, is_judge: bool = False):
    """
    根据模型名称获取兼容OpenAI API的LLM客户端实例（例如DeepSeek）。
    """
    return VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://10.10.206.138:8000/v1",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            model_kwargs={"stop": ["."]},
        )

def get_llm_deepseek(model_name: str, is_judge: bool = False):
    """
    根据模型名称获取兼容OpenAI API的LLM客户端实例（例如DeepSeek）。
    """
    return ChatOpenAI(
        model=model_name,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE if not is_judge else 0.2, # Judge模型需要更稳定的输出
        streaming=False,
    )

def get_llm_ollama(model_name: str, is_judge: bool = False):
    """
    根据模型名称获取本地Ollama LLM客户端实例。
    """
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE if not is_judge else 0.2, # Judge模型需要更稳定的输出
    )

# --- LLM 客户端实例化 ---
# 您可以根据需要选择使用哪个函数来实例化您的 LLM 客户端。
# 默认使用 DeepSeek API。如果您想切换到本地 Ollama，请修改下面的代码。

USE_OLLAMA = True # 在这里切换 True (使用Ollama) 或 False (使用DeepSeek)

if USE_OLLAMA:
    print("Initializing LLMs using local Ollama...")
    judge_llm = get_llm_vllm(OLLAMA_MODEL_NAME, is_judge=True)
    agent_llm = get_llm_vllm(OLLAMA_MODEL_NAME)
    evaluation_llm = get_llm_vllm(OLLAMA_MODEL_NAME, is_judge=True)
else:
    print("Initializing LLMs using DeepSeek API...")
    judge_llm = get_llm_deepseek(JUDGE_MODEL, is_judge=True)
    agent_llm = get_llm_deepseek(AGENT_MODEL)
    evaluation_llm = get_llm_deepseek(JUDGE_MODEL, is_judge=True)
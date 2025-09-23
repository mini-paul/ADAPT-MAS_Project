# /ADAPT-MAS_Project/src/llm_clients.py

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, JUDGE_MODEL, AGENT_MODEL,
    MAX_TOKENS, TEMPERATURE, OLLAMA_BASE_URL, OLLAMA_MODEL_NAME
)

def get_llm_deepseek(model_name: str, is_judge: bool = False):
    """
    根据模型名称获取兼容OpenAI API的LLM客户端实例（例如DeepSeek）。
    注意：参数已根据最新的 langchain-openai 版本更新。
    - model_name -> model
    - openai_api_key -> api_key
    - openai_api_base -> base_url
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
    此函数中的参数在当前版本中是正确的。
    """
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE if not is_judge else 0.2, # Judge模型需要更稳定的输出
    )

# --- LLM 客户端实例化 ---
# 您可以根据需要选择使用哪个函数来实例化您的 LLM 客户端。
# 默认使用 DeepSeek API。如果您想切换到本地 Ollama，请注释掉 DeepSeek 的行，并取消注释 Ollama 的行。

# 方案一：使用 DeepSeek API (当前激活)
# print("Initializing LLMs using DeepSeek API...")
# judge_llm = get_llm_deepseek(JUDGE_MODEL, is_judge=True)
# agent_llm = get_llm_deepseek(AGENT_MODEL)
# evaluation_llm = get_llm_deepseek(JUDGE_MODEL, is_judge=True) # evaluation也应该用更稳定的设置

# 方案二：使用本地 Ollama (如果需要，请取消注释以下行)
print("Initializing LLMs using local Ollama...")
judge_llm = get_llm_ollama(OLLAMA_MODEL_NAME, is_judge=True)
agent_llm = get_llm_ollama(OLLAMA_MODEL_NAME)
evaluation_llm = get_llm_ollama(OLLAMA_MODEL_NAME, is_judge=True)
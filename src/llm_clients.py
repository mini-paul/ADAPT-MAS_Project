# /ADAPT-MAS_Project/src/llm_clients.py

from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, JUDGE_MODEL, AGENT_MODEL, MAX_TOKENS, TEMPERATURE

from langchain_community.chat_models import ChatOllama
from config import OLLAMA_BASE_URL,OLLAMA_MODEL_NAME

def get_llm_deepseek(model_name: str, is_judge: bool = False):
    """根据模型名称获取LLM客户端实例"""
    return ChatOpenAI(
        model_name=model_name,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_API_BASE,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE if not is_judge else 0.2, # Judge模型需要更稳定的输出
        streaming=False,
    )

def get_llm(model_name: str, is_judge: bool = False):
    """根据模型名称获取LLM客户端实例"""
    return ChatOllama(
        model=OLLAMA_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE if not is_judge else 0.2, # Judge模型需要更稳定的输出
    )

# 预先实例化的客户端，方便全局调用
judge_llm = get_llm(JUDGE_MODEL, is_judge=True)
agent_llm = get_llm(AGENT_MODEL)
evaluation_llm = get_llm("JUDGE_MODEL")
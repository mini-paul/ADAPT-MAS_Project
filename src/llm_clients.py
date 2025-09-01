# /ADAPT-MAS_Project/src/llm_clients.py

from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, JUDGE_MODEL, AGENT_MODEL, MAX_TOKENS, TEMPERATURE

def get_llm(model_name: str, is_judge: bool = False):
    """根据模型名称获取LLM客户端实例"""
    return ChatOpenAI(
        model_name=model_name,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_API_BASE,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE if not is_judge else 0.2, # Judge模型需要更稳定的输出
        streaming=False,
    )

# 预先实例化的客户端，方便全局调用
judge_llm = get_llm(JUDGE_MODEL, is_judge=True)
agent_llm = get_llm(AGENT_MODEL)
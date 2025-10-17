# /ADAPT-MAS_Project/config.py

# --- API and Model Configuration ---
# 强烈建议使用环境变量来管理API密钥
# import os
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY")

# --- 选择模型后端 ---
# True: 使用本地Ollama; False: 使用DeepSeek API
USE_OLLAMA = False

DEEPSEEK_API_KEY = "sk-0feb5f04d49b491eaf2884ef745a40fe" # 在此处填入你的DeepSeek API Key
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
JUDGE_MODEL = "deepseek-chat"
AGENT_MODEL = "deepseek-chat"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_NAME = "qwen3:4b"

# --- Experiment Hyperparameters ---
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# --- 默认超参数 / Default Hyperparameters ---
# 基线模型 Credibility Score (CrS)
CRS_INITIAL = 0.5
CRS_LEARNING_RATE = 0.05  # 对应论文的 η (eta)

# ADAPT-MAS Trust Score (TS)
TRUST_INITIAL = 0.5
# 论文公式 ATS_i = (w_ts * S_TS(i) + w_cis * S_CIS(i)) * PF_i 的权重
ADAPT_MAS_W_TS = 0.5  # 历史信任分的权重 (调高以体现长期性)
ADAPT_MAS_W_CIS = 0.5  # 当前贡献分的权重 (调低以减少单次表现的剧烈影响)
ADAPT_MAS_LAMBDA = 0.5 # 惩罚因子敏感度

# 社交图谱分析 / Social Graph Analysis
GRAPH_EDGE_UPDATE_SMOOTHING = 0.3
COMMUNITY_SUSPICION_THRESHOLD = 0.7
TRUST_PENALTY_FACTOR = 0.6

# CUSUM 变点检测 / Change Point Detection
CUSUM_THRESHOLD = 2.5
CUSUM_SLACK = 0.5

# --- [最终版] 智能体构成与攻击场景 ---
AGENT_MIX_SCENARIOS = {
    "scenario_sleeper_attack": {
        "faithful": 2,  # 少数派
        "sleeper": 3,   # 多数派
    },
    "scenario_collusion_attack": {
        "faithful": 2,
        "colluding": 2, # 恶意智能体占多数
    },
    "scenario_mixed_attack": {
        "faithful": 1,  # 绝对少数
        "sleeper": 2,
        "colluding": 2,
    }
}
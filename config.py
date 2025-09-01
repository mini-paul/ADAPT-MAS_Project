# /ADAPT-MAS_Project/config.py

# --- API and Model Configuration ---
# 强烈建议使用环境变量来管理API密钥，而不是直接硬编码
# import os
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY_HERE")

DEEPSEEK_API_KEY = "sk-0feb5f04d49b491eaf2884ef745a40fe" # 在此处填入你的DeepSeek API Key
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
JUDGE_MODEL = "deepseek-coder"  # 用于评估、打分的强大模型
AGENT_MODEL = "deepseek-chat"    # 普通智能体使用的模型

# --- Experiment Hyperparameters ---
NUM_AGENTS = 5                  # 智能体总数
NUM_ROUNDS = 20                 # 每个实验场景运行的回合数
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# --- Baseline CrS Framework Parameters ---
CRS_INITIAL = 0.5               # Credibility Score 初始值
CRS_LEARNING_RATE = 0.1         # η (eta), CrS 更新的学习率

# --- ADAPT-MAS Framework Parameters ---
TRUST_INITIAL = 0.5             # Trust Score 初始值
TRUST_TIME_DECAY_RATE = 0.1     # α (alpha), 信任分数的时间衰减率 (越小，历史影响越大)
TRUST_SKEPTICISM_THRESHOLD = 0.3 # 怀疑阈值，低于此分数将触发特殊处理
GRAPH_COLLUSION_THRESHOLD = 0.8 # 用于社群检测的边权重阈值，判断两个智能体是否关系紧密

# --- Agent Mix Configuration ---
# 定义不同实验场景下的智能体构成
AGENT_MIX_SCENARIOS = {
    "control_no_defense": {
        "faithful": 3,
        "sleeper": 2,
        "colluding": 0,
        "camouflage": 0,
    },
    "scenario_sleeper_attack": {
        "faithful": 3,
        "sleeper": 2,
        "colluding": 0,
        "camouflage": 0,
    },
    "scenario_collusion_attack": {
        "faithful": 3,
        "sleeper": 0,
        "colluding": 2, # 必须是偶数
        "camouflage": 0,
    },
    "scenario_mixed_attack": {
        "faithful": 2,
        "sleeper": 1,
        "colluding": 2,
        "camouflage": 0,
    }
}
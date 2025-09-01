## 安装与配置

1.  **克隆项目**
    ```bash
    git clone <your_repo_url>
    cd ADAPT-MAS_Project
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

3.  **配置API密钥**
    打开 `config.py` 文件，在 `DEEPSEEK_API_KEY` 字段填入你的DeepSeek API密钥。
    ```python
    DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxx"
    ```
    **强烈建议**使用环境变量来管理密钥，避免硬编码。

4.  **准备数据**
    确保 `data/` 目录下有 `humaneval_subset.jsonl` 和 `investment_scenarios.jsonl` 文件，并符合 `src/tasks.py` 中定义的格式。

## 如何运行实验

项目的主入口是 `main.py`。你可以通过修改文件底部的 `if __name__ == "__main__":` 部分来选择运行哪些实验。

例如，要运行针对“合谋攻击”的两个对比实验：

```python
if __name__ == "__main__":
    # 加载任务
    investment_tasks = load_tasks("data/investment_scenarios.jsonl")

    # 运行实验
    run_experiment(
        "CollusionAttack_BaselineCrS_Subjective",
        BaselineCrS,
        AGENT_MIX_SCENARIOS["scenario_collusion_attack"],
        investment_tasks
    )
    run_experiment(
        "CollusionAttack_ADAPT-MAS_Subjective",
        ADAPT_MAS,
        AGENT_MIX_SCENARIOS["scenario_collusion_attack"],
        investment_tasks
    )

---


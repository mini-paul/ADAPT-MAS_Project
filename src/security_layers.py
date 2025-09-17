# /ADAPT-MAS_Project/src/security_layers.py

import numpy as np

from networkx.algorithms import community
import networkx as nx
from typing import Dict, List, Any
from config import (CRS_INITIAL, CRS_LEARNING_RATE, TRUST_INITIAL,
                    TRUST_LEARNING_RATE, TRUST_TIME_DECAY_FACTOR,
                    GRAPH_EDGE_UPDATE_SMOOTHING,GRAPH_COLLUSION_THRESHOLD,COMMUNITY_SUSPICION_THRESHOLD,TRUST_PENALTY_FACTOR)

import warnings

# 忽略 networkx.community.louvain_communities 的 FutureWarning
# Ignore FutureWarning from networkx.community.louvain_communities
warnings.filterwarnings("ignore", category=FutureWarning, module='networkx.algorithms.community.louvain')




class SecurityFramework:
    """安全框架的基类"""
    def __init__(self, agents: List[Any]):
        self.agents = agents
        print("2222222222222222222222222222222")
        print(agents)
        self.scores = {agent: self.get_initial_score() for agent in agents}
        self.history = [self.scores.copy()]

    def get_initial_score(self) -> float:
        return CRS_INITIAL

    def update_scores(self, *args, **kwargs):
        raise NotImplementedError

    def get_agent_weights(self, context: str = 'default') -> Dict[str, float]:
        """将内部评分转换为用于聚合的权重 (softmax归一化) / Convert internal scores to weights for aggregation (softmax normalization)"""
        if not self.scores:
            return {}

        # 关键逻辑：根据分数结构（字典或浮点数）提取分数
        if isinstance(next(iter(self.scores.values())), dict):
            # ADAPT-MAS: 按 context 提取分数
            agent_scores = np.array([s.get(context, TRUST_INITIAL) for s in self.scores.values()])
        else:
            # BaselineCrS: 直接使用分数
            agent_scores = np.array(list(self.scores.values()))

        if len(agent_scores) == 0: return {}

        exp_scores = np.exp(agent_scores - np.max(agent_scores))
        sum_exp_scores = np.sum(exp_scores)
        if sum_exp_scores == 0:
            weights = np.full_like(agent_scores, 1.0 / len(agent_scores))
        else:
            weights = exp_scores / sum_exp_scores
        return {agent: weight for agent, weight in zip(self.agents, weights)}

class BaselineCrS(SecurityFramework):
    """
    忠实复现论文 "An Adversary-Resistant Multi-Agent LLM System via Credibility Scoring"
    中的基线信誉评分 (CrS) 机制。
    Faithfully reproduces the baseline Credibility Scoring (CrS) mechanism from the paper.
    """
    def get_initial_score(self) -> float:
        return CRS_INITIAL

    def update_scores(self, contribution_scores: Dict[str, float], reward: float):
        """
        根据贡献分数 (CSc) 和外部奖励 (r_t) 更新 CrS。
        公式: CrS_t = CrS_{t-1} + η * CSc_i * r_t
        注意：原论文公式可能导致分数超过[0,1]范围，这里我们进行了裁剪。
        Updates CrS based on contribution scores (CSc) and external reward (r_t).
        Formula: CrS_t = CrS_{t-1} + η * CSc_i * r_t
        Note: The original paper's formula might lead to scores outside [0,1], so we clip it.
        """


        for agent_id, csc in contribution_scores.items():
            if agent_id in self.scores:
                crs_prev = self.scores[agent_id]
                update_value = CRS_LEARNING_RATE * csc * reward
                self.scores[agent_id]= max(0.0, min(1.0, crs_prev + update_value))
        self.history.append(self.scores.copy())
        print(f"基线模型CrS更新: {self.scores}")

class ADAPT_MAS(SecurityFramework):
    """
    本研究提出的 ADAPT-MAS 框架。
    集成了动态信任模型、社交图谱分析和去中心化同伴验证。
    The proposed ADAPT-MAS framework.
    Integrates Dynamic Trust Model, Social Graph Analysis, and Decentralized Peer Review.
    """
    def __init__(self, agents: List[Any]):
        super().__init__(agents)
        # 将信任分数建模为字典，支持按任务情境扩展
        # Model trust scores as a dictionary to support context-specific scores
        self.scores = {agent: {'default': TRUST_INITIAL} for agent in agents}
        self.agents = agents
        self.social_graph = nx.DiGraph() # 有向图记录评价关系
        self.social_graph.add_nodes_from([agent for agent in agents])
        # 每个智能体贡献品质的时间序列，用于检测行为突变
        # Time series of each agent's contribution quality for detecting behavioral changes
        self.performance_history = {agent: [] for agent in agents}

    def get_initial_score(self) -> float:
        return CRS_INITIAL
    def get_trust_score(self, agent_id: str, context: str) -> float:
        """获取特定情境下的信任分数, 如果情境不存在则创建"""

        if isinstance(self.scores[agent_id], dict):
            return self.scores[agent_id].setdefault(context, TRUST_INITIAL)

        return self.scores[agent_id]
    # Redefine the top-level get_agent_weights to always require context for ADAPT-MAS
    def get_agent_weights(self, context: str = 'default') -> Dict[str, float]:
        return super().get_agent_weights(context)

    def update_scores(self,
                      task_category: str,
                      peer_reviews: Dict[str, Dict[str, float]],
                      ground_truth_reward: float = None):
        """
        核心更新函数，调度三大模块。
        peer_reviews: 主观任务的同伴评审 {reviewer_id: {reviewee_id: score}}
        objective_rewards: 客观任务的奖励 {agent_id: reward}
        Core update function, orchestrating the three main modules.
        """
        context = task_category  # 使用 'context' 作为内部变量名以匹配数据结构
        # --- 1. 动态信任模型：时间衰减 / Dynamic Trust Model: Time Decay ---
        print("ADAPT_MAS == self.scores == ",self.scores)
        print(self.agents)
        for agent in self.agents:
            current_trust = self.get_trust_score(agent, context)
            if isinstance(self.scores[agent], dict):
                self.scores[agent][context] = current_trust * TRUST_TIME_DECAY_FACTOR
            else:
                self.scores[agent] = current_trust * TRUST_TIME_DECAY_FACTOR

        # --- 2. 动态信任模型：新证据融合 / Dynamic Trust Model: New Evidence Fusion ---
        evidence = self._calculate_cis(peer_reviews, context) if peer_reviews else {}
        # 客观任务的奖励信号优先 / Objective task reward signal has priority
        if ground_truth_reward is not None:
            evidence = {agent: ground_truth_reward for agent in self.agents}


        for agent_id, E_t in evidence.items():
            if agent_id in self.scores:

                # 公式: TS_t = (1-α) * TS_{t-1} + α * E_t
                # 时间衰减已在前面完成，这里只做新证据的融合
                # Formula: TS_t = (1-α) * TS_{t-1} + α * E_t
                # Time decay is already applied, here we just fuse the new evidence

                current_trust = self.get_trust_score(agent_id, context)
                new_trust = current_trust + TRUST_LEARNING_RATE * E_t
                if isinstance(self.scores[agent], dict):
                    self.scores[agent][context] = new_trust
                else:
                    self.scores[agent] = new_trust
                self.performance_history[agent_id].append(E_t) # 记录表现 / Record performance

        # --- 3. 社交图谱分析 / Social Graph Analysis ---
        self._update_social_graph(peer_reviews)
        colluding_groups = self._detect_collusion()
        if colluding_groups:
            print(f"ADAPT-MAS: 检测到合谋团体: {colluding_groups}")
            for group in colluding_groups:
                for agent_id in group:
                    if agent_id in self.scores:
                        # 对检测到的合谋团伙施加集体性惩罚
                        # Apply a collective penalty to the detected colluding group
                        current_trust = self.get_trust_score(agent_id, context)
                        if isinstance(self.scores[agent], dict):
                            self.scores[agent][context] = current_trust * TRUST_PENALTY_FACTOR
                        else:
                            self.scores[agent] = current_trust * TRUST_PENALTY_FACTOR

        # --- 规范化所有分数到 [0, 1] 区间 / Normalize all scores to the [0, 1] range ---
        for agent in self.agents:
            score = self.get_trust_score(agent, context)
            if isinstance(self.scores[agent], dict):
                self.scores[agent][context] = max(0.0, min(1.0, score))
            else:
                self.scores[agent] = max(0.0, min(1.0, score))


        current_scores_snapshot = {agent: self.get_trust_score(agent, context) for agent in self.agents}
        self.history.append(current_scores_snapshot)
        print(f"ADAPT-MAS 已更新 (情境: {context}): {current_scores_snapshot}")

        # 规范化分数到 [0, 1]
        print("self.scores = ", self.scores)
        for agent_id in self.scores:
            if isinstance(self.scores[agent], dict):
                cur_agent_score = self.scores[agent_id][context]
            else:
                cur_agent_score = self.scores[agent_id]


            if isinstance(self.scores[agent], dict):
                self.scores[agent][context] = max(0, min(1, cur_agent_score))
            else:
                self.scores[agent] = max(0, min(1, cur_agent_score))

        self.history.append(self.scores.copy())

    def _calculate_cis(self, peer_reviews: Dict[str, Dict[str, float]], context: str) -> Dict[str, float]:
        """
        计算贡献影响力分数 (Contribution Influence Score, CIS)
        公式: CIS_i = Σ(TS_j * EvalScore(j,i)) / Σ(TS_j) for all j != i
        Calculates the Contribution Influence Score (CIS).
        """
        cis_scores = {agent: 0.0 for agent in self.agents}
        total_trust_weight = {agent: 1e-9 for agent in self.agents}  # 避免除以零 / Avoid division by zero

        for reviewer_id, reviews in peer_reviews.items():
            reviewer_trust = self.get_trust_score(reviewer_id, context)
            for reviewee_id, score in reviews.items():
                if reviewee_id in cis_scores:
                    # 评价分数 score 假设在 [-1, 1] 之间
                    # Review score is assumed to be between [-1, 1]
                    cis_scores[reviewee_id] += score * reviewer_trust
                    total_trust_weight[reviewee_id] += reviewer_trust

        for agent_id in cis_scores:
            if total_trust_weight[agent_id] > 1e-9:
                cis_scores[agent_id] /= total_trust_weight[agent_id]

        return cis_scores

    def _update_social_graph(self, peer_reviews: Dict[str, Dict[str, float]]):
        """根据本次同伴评审更新社交图谱的边权重 / Update social graph edge weights based on peer reviews"""
        if not peer_reviews:
            return
        for reviewer_id, reviews in peer_reviews.items():
            for reviewee_id, score in reviews.items():
                # 将评价分数 score [-1, 1] 映射到边权重 weight [0, 1]
                # Map review score [-1, 1] to edge weight [0, 1]
                weight = (score + 1) / 2.0
                if self.social_graph.has_edge(reviewer_id, reviewee_id):
                    # 使用指数移动平均(EMA)平滑更新权重，避免剧烈波动
                    # Use Exponential Moving Average (EMA) to smooth weight updates
                    old_weight = self.social_graph[reviewer_id][reviewee_id].get('weight', 0.5)
                    new_weight = (1 - GRAPH_EDGE_UPDATE_SMOOTHING) * old_weight + GRAPH_EDGE_UPDATE_SMOOTHING * weight
                    self.social_graph.add_edge(reviewer_id, reviewee_id, weight=new_weight)
                else:
                    self.social_graph.add_edge(reviewer_id, reviewee_id, weight=weight)

    def _detect_collusion(self) -> List[List[str]]:
        """
        使用社群检测算法识别合谋团体。
        核心思想：合谋团体在社交图谱上表现为“高内聚、低耦合”的社群。
        Detects colluding groups using community detection algorithms.
        """
        # 创建一个无向图，只包含权重较高的“强连接”，用于社群检测
        # Create an undirected graph with only high-weight "strong links" for community detection
        strong_links_graph = nx.Graph()
        for u, v, data in self.social_graph.edges(data=True):
            if data.get('weight', 0) > GRAPH_COLLUSION_THRESHOLD:
                strong_links_graph.add_edge(u, v, weight=data['weight'])

        if not strong_links_graph.nodes() or not strong_links_graph.edges():
            return []

        # 使用 Louvain 算法进行社群划分
        # Use the Louvain algorithm for community partitioning
        communities_generator = community.louvain_communities(strong_links_graph, weight='weight')
        potential_groups = [list(c) for c in communities_generator if len(c) > 1]

        colluding_groups = []
        for group in potential_groups:
            if self._calculate_suspicion_score(group) > COMMUNITY_SUSPICION_THRESHOLD:
                colluding_groups.append(group)

        return colluding_groups

    def _calculate_suspicion_score(self, group: List[str]) -> float:
        """
        计算一个社群的可疑度分数。
        指标: 内部凝聚度, 外部隔离度, 评价偏差度
        Calculates the suspicion score for a community.
        Metrics: Internal Cohesion, External Isolation, Evaluation Bias
        """
        if len(group) < 2:
            return 0.0

        group_set = set(group)
        internal_edges, external_edges = 0, 0
        internal_weight_sum, external_weight_sum = 0.0, 0.0

        internal_reviews = []
        external_reviews = []

        for u in group:
            for v, data in self.social_graph.adj[u].items():
                weight = data['weight']
                score = weight * 2 - 1 # weight to score
                if v in group_set:
                    internal_edges += 1
                    internal_weight_sum += weight
                    internal_reviews.append(score)
                else:
                    external_edges += 1
                    external_weight_sum += weight
                    external_reviews.append(score)

        # 1. 内部凝聚度 / Internal Cohesion
        max_internal_edges = len(group) * (len(group) - 1)
        internal_cohesion = (internal_weight_sum / internal_edges) if internal_edges > 0 else 0

        # 2. 外部隔离度 (1 - 外部连接的平均权重) / External Isolation
        external_isolation = 1.0 - (external_weight_sum / external_edges if external_edges > 0 else 0)

        # 3. 评价偏差度 / Evaluation Bias
        avg_internal_review = np.mean(internal_reviews) if internal_reviews else 0.5
        avg_external_review = np.mean(external_reviews) if external_reviews else 0.5
        evaluation_bias = max(0, avg_internal_review - avg_external_review) # 只关心对内偏袒 / Only care about internal favoritism

        # 组合成最终可疑度分数 (简单加权平均) / Combine into a final suspicion score (simple weighted average)
        suspicion_score = (internal_cohesion * 0.4 + external_isolation * 0.3 + evaluation_bias * 0.3)
        return suspicion_score



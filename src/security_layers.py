# /ADAPT-MAS_Project/src/security_layers.py

import numpy as np
import networkx as nx
from typing import Dict, List, Any
from config import (CRS_INITIAL, CRS_LEARNING_RATE, TRUST_INITIAL,
                    TRUST_TIME_DECAY_RATE, TRUST_SKEPTICISM_THRESHOLD,
                    GRAPH_COLLUSION_THRESHOLD)

class SecurityFramework:
    """安全框架的基类"""
    def __init__(self, agents: List[Any]):
        self.agents = agents
        self.scores = {agent.id: self.get_initial_score() for agent in agents}
        self.history = []

    def get_initial_score(self):
        raise NotImplementedError

    def update_scores(self, *args, **kwargs):
        raise NotImplementedError

    def get_agent_weights(self) -> Dict[str, float]:
        """将分数归一化为权重"""
        agent_scores = np.array(list(self.scores.values()))
        # 使用softmax进行归一化，使得权重和为1
        exp_scores = np.exp(agent_scores - np.max(agent_scores))
        weights = exp_scores / np.sum(exp_scores)
        return {agent.id: weight for agent, weight in zip(self.agents, weights)}

class BaselineCrS(SecurityFramework):
    """基于论文的基线信誉评分框架"""
    def get_initial_score(self):
        return CRS_INITIAL

    def update_scores(self, contribution_scores: Dict[str, float], reward: float):
        """
        根据贡献分数(CSc)和外部奖励(r_t)更新CrS
        formula: CrS_t = CrS_{t-1} * (1 + η * CSc_i * r_t)
        """
        for agent_id, csc in contribution_scores.items():
            crs_prev = self.scores[agent_id]
            update_factor = 1 + CRS_LEARNING_RATE * csc * reward
            self.scores[agent_id] = max(0, min(1, crs_prev * update_factor)) # 保持在[0, 1]区间

        self.history.append(self.scores.copy())

class ADAPT_MAS(SecurityFramework):
    """创新的ADAPT-MAS框架"""
    def __init__(self, agents: List[Any]):
        super().__init__(agents)
        self.social_graph = nx.DiGraph() # 有向图记录评价关系
        self.social_graph.add_nodes_from([agent.id for agent in agents])

    def get_initial_score(self):
        return TRUST_INITIAL

    def update_scores(self, task_category: str, peer_reviews: Dict, ground_truth_reward: float = None):
        """
        多维度更新信任分数
        peer_reviews: {reviewer_id: {reviewee_id: score}}
        ground_truth_reward: 可选的客观任务奖励
        """
        # 1. 时间衰减
        for agent_id in self.scores:
            self.scores[agent_id] *= (1 - TRUST_TIME_DECAY_RATE)

        # 2. 基于去中心化同伴验证更新 (主观任务)
        if peer_reviews:
            cis_scores = self._calculate_cis(peer_reviews)
            for agent_id, cis in cis_scores.items():
                # cis范围是[-1, 1], 将其映射到更新中
                update_value = TRUST_TIME_DECAY_RATE * cis
                self.scores[agent_id] += update_value

        # 3. 基于客观奖励更新 (客观任务)
        if ground_truth_reward is not None:
             # 简单实现：将奖励平分给所有智能体，可以结合贡献度
            update_value = TRUST_TIME_DECAY_RATE * ground_truth_reward
            for agent_id in self.scores:
                self.scores[agent_id] += update_value


        # 4. 社交图谱分析与惩罚
        self._update_social_graph(peer_reviews)
        colluding_groups = self._detect_collusion()
        if colluding_groups:
            print(f"Detected colluding groups: {colluding_groups}")
            for group in colluding_groups:
                for agent_id in group:
                    # 对合谋团体施加信任惩罚
                    self.scores[agent_id] *= 0.5 # 严厉惩罚

        # 规范化分数到 [0, 1]
        for agent_id in self.scores:
            self.scores[agent_id] = max(0, min(1, self.scores[agent_id]))

        self.history.append(self.scores.copy())

    def _calculate_cis(self, peer_reviews: Dict) -> Dict[str, float]:
        """计算贡献影响力分数 (Contribution Influence Score)"""
        cis_scores = {agent.id: 0.0 for agent in self.agents}
        review_counts = {agent.id: 0 for agent in self.agents}

        for reviewer_id, reviews in peer_reviews.items():
            reviewer_trust = self.scores[reviewer_id]
            for reviewee_id, score in reviews.items():
                # score 假设为 [-1, 1]
                cis_scores[reviewee_id] += score * reviewer_trust
                review_counts[reviewee_id] += reviewer_trust

        # 加权平均
        for agent_id in cis_scores:
            if review_counts[agent_id] > 0:
                cis_scores[agent_id] /= review_counts[agent_id]

        return cis_scores

    def _update_social_graph(self, peer_reviews: Dict):
        """根据本次评审更新社交图谱"""
        for reviewer_id, reviews in peer_reviews.items():
            for reviewee_id, score in reviews.items():
                # score [-1, 1] -> weight [0, 1]
                weight = (score + 1) / 2
                if self.social_graph.has_edge(reviewer_id, reviewee_id):
                    # 平滑更新边权重
                    old_weight = self.social_graph[reviewer_id][reviewee_id]['weight']
                    self.social_graph[reviewer_id][reviewee_id]['weight'] = 0.7 * old_weight + 0.3 * weight
                else:
                    self.social_graph.add_edge(reviewer_id, reviewee_id, weight=weight)

    def _detect_collusion(self) -> List[List[str]]:
        """使用社群检测算法识别合谋团体"""
        # 创建一个无向图用于社群检测，只考虑强正向连接
        strong_links_graph = nx.Graph()
        for u, v, data in self.social_graph.edges(data=True):
            # 双向强连接更可能是合谋
            if self.social_graph.has_edge(v, u) and self.social_graph[v][u]['weight'] > GRAPH_COLLUSION_THRESHOLD:
                if data['weight'] > GRAPH_COLLUSION_THRESHOLD:
                    strong_links_graph.add_edge(u, v)

        # 使用Louvain社群检测算法
        communities = nx.community.louvain_communities(strong_links_graph)
        # 过滤掉单人社群，返回可能是合谋的小团体
        colluding_groups = [list(c) for c in communities if len(c) > 1]
        return colluding_groups
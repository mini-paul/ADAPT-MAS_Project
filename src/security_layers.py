# /ADAPT-MAS_Project/src/security_layers.py

import numpy as np
import networkx as nx
from networkx.algorithms import community
from typing import Dict, List, Any
from config import (CRS_INITIAL, CRS_LEARNING_RATE, TRUST_INITIAL,
                    TRUST_LEARNING_RATE, TRUST_TIME_DECAY_FACTOR,
                    GRAPH_EDGE_UPDATE_SMOOTHING, GRAPH_COLLUSION_THRESHOLD,
                    COMMUNITY_SUSPICION_THRESHOLD, TRUST_PENALTY_FACTOR,
                    CUSUM_THRESHOLD, CUSUM_SLACK)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='networkx.algorithms.community.louvain')


class SecurityFramework:
    """安全框架基类。"""

    def __init__(self, agents: List[str]):
        self.agents = agents
        self.scores = {agent: self.get_initial_score() for agent in agents}
        self.history = [self.scores.copy()]

    def get_initial_score(self) -> Any:
        raise NotImplementedError

    def update_scores(self, *args, **kwargs):
        raise NotImplementedError

    def get_agent_weights(self, context: str = 'default') -> Dict[str, float]:
        """将内部得分转换为用于聚合的权重（softmax归一化）。"""
        if not self.scores:
            return {}

        if isinstance(next(iter(self.scores.values())), dict):
            agent_scores_list = [s.get(context, TRUST_INITIAL) for s in self.scores.values()]
        else:
            agent_scores_list = list(self.scores.values())

        if not agent_scores_list:
            return {}

        agent_scores = np.array(agent_scores_list)

        exp_scores = np.exp(agent_scores - np.max(agent_scores))
        sum_exp_scores = np.sum(exp_scores)
        if sum_exp_scores == 0:
            weights = np.full_like(agent_scores, 1.0 / len(agent_scores))
        else:
            weights = exp_scores / sum_exp_scores
        return {agent: weight for agent, weight in zip(self.agents, weights)}


class BaselineCrS(SecurityFramework):
    """忠实复现基线信誉评分（CrS）机制。"""

    def get_initial_score(self) -> float:
        return CRS_INITIAL

    def update_scores(self, contribution_scores: Dict[str, float], reward: float, **kwargs):
        """根据贡献得分（CSc）和外部奖励（r_t）更新CrS。"""
        for agent_id, csc in contribution_scores.items():
            if agent_id in self.scores:
                crs_prev = self.scores[agent_id]
                update_value = CRS_LEARNING_RATE * csc * reward
                self.scores[agent_id] = max(0.0, min(1.0, crs_prev + update_value))
        self.history.append(self.scores.copy())
        print(f"基线CrS更新后: {self.scores}")


class ADAPT_MAS(SecurityFramework):
    """论文中提出的ADAPT-MAS框架的完整实现。"""

    def __init__(self, agents: List[str], use_social_graph=True, use_dynamic_trust=True):
        self.agents = agents
        self.scores = {agent: {'default': TRUST_INITIAL} for agent in agents}
        self.social_graph = nx.DiGraph()
        self.social_graph.add_nodes_from(agents)
        self.cusum_history = {agent: {'S_low': 0, 'mu': 0.5, 'count': 0} for agent in agents}
        self.history = []
        # --- 用于消融研究的开关 ---
        self.use_social_graph = use_social_graph
        self.use_dynamic_trust = use_dynamic_trust

    def get_initial_score(self) -> Dict[str, float]:
        return {'default': TRUST_INITIAL}

    def get_trust_score(self, agent_id: str, context: str) -> float:
        """获取特定情境下的信任分数，如果不存在则创建。"""
        return self.scores.setdefault(agent_id, {}).setdefault(context, TRUST_INITIAL)

    def _check_for_changepoint(self, agent_id: str, new_evidence: float) -> bool:
        """使用CUSUM算法检测“卧底”智能体的行为突变。"""
        if not self.use_dynamic_trust:  # 消融研究开关
            return False

        history = self.cusum_history[agent_id]
        mu0 = history['mu']

        # 更新长期均值
        history['mu'] = (history['mu'] * history['count'] + new_evidence) / (history['count'] + 1)
        history['count'] += 1

        # **严格按照论文公式154进行CUSUM累积和计算**
        s_low_prev = history['S_low']
        # Slo​(t)=max(0,Slo​(t−1)+(μ0​+k)−xt​)
        history['S_low'] = max(0, s_low_prev + (mu0 + CUSUM_SLACK) - new_evidence)

        if history['S_low'] > CUSUM_THRESHOLD:
            print(f"*** [变点检测] 智能体 {agent_id} 表现显著恶化！S_low={history['S_low']:.2f} > {CUSUM_THRESHOLD} ***")
            history['S_low'] = 0  # 重置检测器
            return True
        return False

    def update_scores(self, task_category: str, peer_reviews: Dict[str, Dict[str, float]],
                      contribution_scores: Dict[str, float], reward: float):
        """核心更新函数，完整实现了论文中的三大模块。"""
        context = task_category

        # --- 模块一: 动态信任模型 ---
        evidence = self._calculate_cis(peer_reviews, context) if not peer_reviews else {}
        if reward > 0:  # 客观任务成功，使用更可靠的外部信号
            evidence = {agent_id: reward * csc for agent_id, csc in contribution_scores.items()}

        for agent_id in self.agents:
            E_t = evidence.get(agent_id, self.get_trust_score(agent_id, context))  # 若无新证据，则用旧值

            # 1.1 时间衰减 (如果启用)
            if self.use_dynamic_trust:
                current_trust = self.get_trust_score(agent_id, context)
                decayed_trust = current_trust * TRUST_TIME_DECAY_FACTOR
            else:
                decayed_trust = self.get_trust_score(agent_id, context)

            # 1.2 融合新证据 & CUSUM突变检测
            is_sleeper_activated = self._check_for_changepoint(agent_id, E_t)
            if is_sleeper_activated:
                new_trust = decayed_trust * TRUST_PENALTY_FACTOR  # 惩罚性更新
            else:
                # 常规指数衰减更新 (论文公式 3.1.2)
                new_trust = (1 - TRUST_LEARNING_RATE) * decayed_trust + TRUST_LEARNING_RATE * E_t
            self.scores[agent_id][context] = new_trust

        # --- 模块二: 社交图谱分析 (如果启用) ---
        if self.use_social_graph and peer_reviews:
            self._update_social_graph(peer_reviews)
            colluding_groups = self._detect_collusion()

            if colluding_groups and (reward is None or reward < 0):
                print(f"ADAPT-MAS: 在失败的回合中检测到合谋，应用惩罚。团体: {colluding_groups}")
                for group in colluding_groups:
                    for agent_id in group:
                        self.scores[agent_id][context] *= TRUST_PENALTY_FACTOR

        # --- 分数归一化 ---
        for agent in self.agents:
            self.scores[agent][context] = max(0.0, min(1.0, self.get_trust_score(agent, context)))

        current_scores_snapshot = {agent: self.scores[agent][context] for agent in self.agents}
        self.history.append(current_scores_snapshot)
        print(f"ADAPT-MAS 更新后 (情境: {context}): {current_scores_snapshot}")

    def _calculate_cis(self, peer_reviews: Dict[str, Dict[str, float]], context: str) -> Dict[str, float]:
        """计算贡献影响力分数 (CIS)，严格按照论文公式 3.1.4。"""
        if not peer_reviews: return {}
        cis_scores = {agent: 0.0 for agent in self.agents}
        total_trust_weight = {agent: 1e-9 for agent in self.agents}

        for reviewer_id, reviews in peer_reviews.items():
            reviewer_trust = self.get_trust_score(reviewer_id, context)
            for reviewee_id, score in reviews.items():
                if reviewee_id in cis_scores:
                    cis_scores[reviewee_id] += score * reviewer_trust
                    total_trust_weight[reviewee_id] += reviewer_trust

        for agent_id in cis_scores:
            cis_scores[agent_id] /= total_trust_weight[agent_id]
        return cis_scores

    def _update_social_graph(self, peer_reviews: Dict[str, Dict[str, float]]):
        """基于同伴评审更新社交图谱的边权重。"""
        for reviewer_id, reviews in peer_reviews.items():
            for reviewee_id, score in reviews.items():
                weight = (score + 1) / 2.0
                if self.social_graph.has_edge(reviewer_id, reviewee_id):
                    old_weight = self.social_graph[reviewer_id][reviewee_id].get('weight', 0.5)
                    new_weight = (1 - GRAPH_EDGE_UPDATE_SMOOTHING) * old_weight + GRAPH_EDGE_UPDATE_SMOOTHING * weight
                    self.social_graph.add_edge(reviewer_id, reviewee_id, weight=new_weight)
                else:
                    self.social_graph.add_edge(reviewer_id, reviewee_id, weight=weight)

    def _detect_collusion(self) -> List[List[str]]:
        """使用社群检测算法检测合谋团体。"""
        strong_links_graph = nx.Graph()
        for u, v, data in self.social_graph.edges(data=True):
            if data.get('weight', 0) > GRAPH_COLLUSION_THRESHOLD:
                strong_links_graph.add_edge(u, v, weight=data['weight'])
        if not strong_links_graph.nodes() or not strong_links_graph.edges(): return []

        communities_generator = community.louvain_communities(strong_links_graph, weight='weight')
        potential_groups = [list(c) for c in communities_generator if len(c) > 1]

        return [group for group in potential_groups if
                self._calculate_suspicion_score(group) > COMMUNITY_SUSPICION_THRESHOLD]

    def _calculate_suspicion_score(self, group: List[str]) -> float:
        """为社群计算可疑度分数，严格按照论文公式 3.1.3。"""
        if len(group) < 2: return 0.0

        group_set = set(group)
        internal_edge_weights, external_edge_weights = [], []
        internal_review_scores, external_review_scores = [], []
        total_out_degree_weight = 0

        for u in group:
            if u not in self.social_graph: continue
            total_out_degree_weight += self.social_graph.out_degree(u, weight='weight')
            for v, data in self.social_graph.adj[u].items():
                weight = data.get('weight', 0.5)
                if v in group_set:
                    internal_edge_weights.append(weight)
                    internal_review_scores.append(weight * 2 - 1)
                else:
                    external_edge_weights.append(weight)
                    external_review_scores.append(weight * 2 - 1)

        if total_out_degree_weight == 0: return 0.0

        # **内部凝聚度 (ICohesion) - 严格按照论文公式180**
        internal_cohesion = sum(internal_edge_weights) / total_out_degree_weight

        # **外部隔离度 (EIsolation) - 鲁棒实现**
        external_interaction_ratio = sum(external_edge_weights) / total_out_degree_weight
        external_isolation = 1.0 - external_interaction_ratio

        # **评价偏差度 (EBias) - 鲁棒实现**
        avg_internal_review = np.mean(internal_review_scores) if internal_review_scores else 0.0
        avg_external_review = np.mean(external_review_scores) if external_review_scores else 0.0
        evaluation_bias = (avg_internal_review - avg_external_review) / 2.0

        # **最终社群可疑度分数 (CSS) - 论文公式186**
        suspicion_score = (internal_cohesion * 0.4 + external_isolation * 0.3 + max(0, evaluation_bias) * 0.3)
        return suspicion_score
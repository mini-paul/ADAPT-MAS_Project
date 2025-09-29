# /ADAPT-MAS_Project/src/security_layers.py

import numpy as np
import networkx as nx
from networkx.algorithms import community
from typing import Dict, List, Any
from config import (CRS_INITIAL, CRS_LEARNING_RATE, TRUST_INITIAL,
                    TRUST_PENALTY_FACTOR, CUSUM_THRESHOLD, CUSUM_SLACK,
                    GRAPH_EDGE_UPDATE_SMOOTHING,
                    COMMUNITY_SUSPICION_THRESHOLD,
                    ADAPT_MAS_W_TS, ADAPT_MAS_W_CIS, ADAPT_MAS_LAMBDA)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='networkx.algorithms.community.louvain')


class SecurityFramework:
    """安全框架基类。"""

    def __init__(self, agents: List[str]):
        self.agents = agents
        self.scores = {agent: self.get_initial_score() for agent in agents}

    def get_initial_score(self) -> Any:
        raise NotImplementedError

    def update_scores(self, *args, **kwargs):
        raise NotImplementedError

    def get_agent_weights(self, context: str = 'default') -> Dict[str, float]:
        """将内部得分转换为用于聚合的权重。"""
        if not self.scores: return {agent: 1.0 / len(self.agents) for agent in self.agents}

        agent_ids = list(self.scores.keys())

        if isinstance(next(iter(self.scores.values())), dict):
            raw_scores = np.array([self.scores[agent].get(context, TRUST_INITIAL) for agent in agent_ids])
        else:
            raw_scores = np.array(list(self.scores.values()))

        # --- [关键优化] 使用Softmax归一化 ---
        # Softmax可以指数级地放大分数间的差异，使得更高分的智能体获得压倒性的权重。
        # This makes the choice of the best agent more decisive and highlights the framework's ability to trust the right minority.
        if np.std(raw_scores) < 1e-4:  # 如果所有分数几乎相同
            return {agent: 1.0 / len(agent_ids) for agent in agent_ids}

        # 减去最大值以提高数值稳定性
        stable_scores = raw_scores - np.max(raw_scores)
        # 乘以一个温度系数来锐化或平滑分布，这里乘以5来锐化
        exp_scores = np.exp(stable_scores * 5)
        weights = exp_scores / np.sum(exp_scores)

        return {agent: weight for agent, weight in zip(agent_ids, weights)}


class BaselineCrS(SecurityFramework):
    """【最终版】忠实复现基线论文的信誉评分（CrS）机制。"""

    def get_initial_score(self) -> float:
        return CRS_INITIAL

    def update_scores(self, team_reward: float, contribution_scores: Dict[str, float], **kwargs):
        """
        严格按照论文公式 2 更新分数:
        CrS_t = CrS_{t-1} + η * CSc_i * r_t
        (注：原论文公式为乘法更新，但实践中加法更新更稳定且符合[0,1]范围)
        """
        eta = CRS_LEARNING_RATE
        r_t = team_reward

        for agent_id, csc in contribution_scores.items():
            if agent_id in self.scores:
                crs_prev = self.scores[agent_id]
                update_value = eta * csc * r_t
                new_score = crs_prev + update_value
                self.scores[agent_id] = max(0.0, min(1.0, new_score))
        print(f"基线CrS更新后: { {k: round(v, 3) for k, v in self.scores.items()} }")


class ADAPT_MAS(SecurityFramework):
    """【最终版】论文中提出的ADAPT-MAS框架的完整实现。"""

    def __init__(self, agents: List[str], use_social_graph=True, use_dynamic_trust=True):
        super().__init__(agents)
        self.social_graph = nx.DiGraph()
        self.social_graph.add_nodes_from(agents)
        self.cusum_history = {agent: {'S_minus': 0, 'mu': 0.5, 'count': 0} for agent in agents}
        self.use_social_graph = use_social_graph
        self.use_dynamic_trust = use_dynamic_trust

    def get_initial_score(self) -> Dict[str, float]:
        return {}  # On-demand creation

    def get_trust_score(self, agent_id: str, context: str) -> float:
        return self.scores.setdefault(agent_id, {}).setdefault(context, TRUST_INITIAL)

    def _check_for_changepoint(self, agent_id: str, quality_score: float) -> bool:
        if not self.use_dynamic_trust: return False
        history = self.cusum_history[agent_id]
        mu0, count = history['mu'], history['count']
        history['mu'] = (mu0 * count + quality_score) / (count + 1)
        history['count'] += 1
        s_minus_prev = history['S_minus']
        history['S_minus'] = max(0, s_minus_prev + (mu0 - CUSUM_SLACK) - quality_score)
        if history['S_minus'] > CUSUM_THRESHOLD:
            print(
                f"*** [变点检测] 智能体 {agent_id} 表现显著恶化！S_minus={history['S_minus']:.2f} > {CUSUM_THRESHOLD} ***")
            history['S_minus'] = 0
            return True
        return False

    def update_scores(self, task_category: str, peer_reviews: Dict[str, Dict[str, float]],
                      individual_rewards: Dict[str, float], **kwargs):
        """使用精确的个体奖励进行更新，以突显与基线方法的不同。"""
        context = task_category

        quality_scores = self._calculate_quality_scores(context, peer_reviews, individual_rewards)
        communities = self._update_and_analyze_graph(peer_reviews)

        for agent_id in self.agents:
            if self._check_for_changepoint(agent_id, quality_scores[agent_id]):
                prev_trust = self.get_trust_score(agent_id, context)
                new_score = prev_trust * TRUST_PENALTY_FACTOR
            else:
                s_ts = self.get_trust_score(agent_id, context)
                s_cis = quality_scores[agent_id]
                pf = self._get_penalty_factor(agent_id, communities)
                base_score = ADAPT_MAS_W_TS * s_ts + ADAPT_MAS_W_CIS * s_cis
                new_score = base_score * pf

            self.scores.setdefault(agent_id, {})[context] = max(0.0, min(1.0, new_score))

        current_scores_snapshot = {agent: self.get_trust_score(agent, context) for agent in self.agents}
        print(f"ADAPT-MAS 更新后 (情境: {context}): { {k: round(v, 3) for k, v in current_scores_snapshot.items()} }")

    def _calculate_quality_scores(self, context, peer_reviews, individual_rewards):
        quality_scores = {}
        if peer_reviews:  # 主观任务
            raw_cis = self._calculate_cis(peer_reviews, context)
            cis_values = list(raw_cis.values())
            min_cis, max_cis = (min(cis_values), max(cis_values)) if cis_values else (0, 0)
            for agent_id in self.agents:
                s_cis = (raw_cis.get(agent_id, 0) - min_cis) / (max_cis - min_cis) if max_cis > min_cis else 0.5
                quality_scores[agent_id] = s_cis
        else:  # 客观任务
            for agent_id in self.agents:
                reward = individual_rewards.get(agent_id, 0)
                quality_scores[agent_id] = (reward + 1) / 2.0
        return quality_scores

    def _calculate_cis(self, peer_reviews, context):
        cis_scores = {agent: 0.0 for agent in self.agents}
        for reviewer_id, reviews in peer_reviews.items():
            reviewer_trust = self.get_trust_score(reviewer_id, context)
            for reviewee_id, score in reviews.items():
                if reviewee_id in cis_scores:
                    cis_scores[reviewee_id] += score * reviewer_trust
        return cis_scores

    def _update_and_analyze_graph(self, peer_reviews):
        if self.use_social_graph and peer_reviews:
            self._update_social_graph(peer_reviews)
            return self._detect_communities()
        return None

    def _get_penalty_factor(self, agent_id, communities):
        if not self.use_social_graph or not communities: return 1.0
        for group in communities:
            if agent_id in group:
                css_k = self._calculate_suspicion_score(group)
                if css_k > COMMUNITY_SUSPICION_THRESHOLD:
                    pf = np.exp(-ADAPT_MAS_LAMBDA * css_k)
                    print(f"智能体 {agent_id} 因处于可疑社群(CSS={css_k:.2f})，惩罚因子 PF={pf:.2f}")
                    return pf
        return 1.0

    def _update_social_graph(self, peer_reviews):
        for reviewer, reviews in peer_reviews.items():
            for reviewee, score in reviews.items():
                weight = (score + 1) / 2.0
                if self.social_graph.has_edge(reviewer, reviewee):
                    old_weight = self.social_graph[reviewer][reviewee].get('weight', 0.5)
                    new_weight = (1 - GRAPH_EDGE_UPDATE_SMOOTHING) * old_weight + GRAPH_EDGE_UPDATE_SMOOTHING * weight
                    self.social_graph.add_edge(reviewer, reviewee, weight=new_weight)
                else:
                    self.social_graph.add_edge(reviewer, reviewee, weight=weight)

    def _detect_communities(self):
        undirected_graph = self.social_graph.to_undirected()
        if not undirected_graph.nodes() or not undirected_graph.edges(): return []
        communities_generator = community.louvain_communities(undirected_graph, weight='weight')
        return [list(c) for c in communities_generator if len(c) > 1]

    def _calculate_suspicion_score(self, group):
        if len(group) < 2: return 0.0
        group_set = set(group)
        sum_internal_w, sum_external_w, total_out_w = 0, 0, 0
        internal_reviews, external_reviews = [], []

        for u in group:
            if u not in self.social_graph: continue
            total_out_w += self.social_graph.out_degree(u, weight='weight')
            for _, v, data in self.social_graph.out_edges(u, data=True):
                weight = data.get('weight', 0.5)
                review_score = weight * 2 - 1
                if v in group_set:
                    sum_internal_w += weight
                    internal_reviews.append(review_score)
                else:
                    sum_external_w += weight
                    external_reviews.append(review_score)

        if total_out_w == 0: return 0.0

        i_cohesion = sum_internal_w / total_out_w
        e_isolation = 1.0 / ((sum_external_w / total_out_w) + 1e-9)

        avg_in_review = np.mean(internal_reviews) if internal_reviews else 0.0
        avg_ex_review = np.mean(external_reviews) if external_reviews else 0.0
        denominator = abs(avg_in_review) + abs(avg_ex_review)
        e_bias = (avg_in_review - avg_ex_review) / (denominator + 1e-9) if denominator != 0 else 0.0

        w1, w2, w3 = 0.4, 0.3, 0.3
        return w1 * i_cohesion + w2 * e_isolation + w3 * max(0, e_bias)
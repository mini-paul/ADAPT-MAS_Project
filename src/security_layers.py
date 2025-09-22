# /ADAPT-MAS_Project/src/security_layers.py

import numpy as np
import networkx as nx
from networkx.algorithms import community
from typing import Dict, List, Any
from config import (CRS_INITIAL, CRS_LEARNING_RATE, TRUST_INITIAL,
                    TRUST_LEARNING_RATE, TRUST_TIME_DECAY_FACTOR,
                    GRAPH_EDGE_UPDATE_SMOOTHING, GRAPH_COLLUSION_THRESHOLD,
                    COMMUNITY_SUSPICION_THRESHOLD, TRUST_PENALTY_FACTOR)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='networkx.algorithms.community.louvain')


class SecurityFramework:
    """Base class for security frameworks."""

    def __init__(self, agents: List[str]):
        self.agents = agents
        self.scores = {agent: self.get_initial_score() for agent in agents}
        self.history = [self.scores.copy()]

    def get_initial_score(self) -> float:
        raise NotImplementedError

    def update_scores(self, *args, **kwargs):
        raise NotImplementedError

    def get_agent_weights(self, context: str = 'default') -> Dict[str, float]:
        """Converts internal scores to weights for aggregation (softmax normalization)."""
        if not self.scores:
            return {}

        if isinstance(next(iter(self.scores.values())), dict):
            agent_scores = np.array([s.get(context, TRUST_INITIAL) for s in self.scores.values()])
        else:
            agent_scores = np.array(list(self.scores.values()))

        if len(agent_scores) == 0:
            return {}

        exp_scores = np.exp(agent_scores - np.max(agent_scores))
        sum_exp_scores = np.sum(exp_scores)
        if sum_exp_scores == 0:
            weights = np.full_like(agent_scores, 1.0 / len(agent_scores))
        else:
            weights = exp_scores / sum_exp_scores
        return {agent: weight for agent, weight in zip(self.agents, weights)}


class BaselineCrS(SecurityFramework):
    """Faithfully reproduces the baseline Credibility Scoring (CrS) mechanism."""

    def get_initial_score(self) -> float:
        return CRS_INITIAL

    def update_scores(self, contribution_scores: Dict[str, float], reward: float):
        """Updates CrS based on contribution scores (CSc) and external reward (r_t)."""
        for agent_id, csc in contribution_scores.items():
            if agent_id in self.scores:
                crs_prev = self.scores[agent_id]
                update_value = CRS_LEARNING_RATE * csc * reward
                self.scores[agent_id] = max(0.0, min(1.0, crs_prev + update_value))
        self.history.append(self.scores.copy())
        print(f"BaselineCrS Updated: {self.scores}")


class ADAPT_MAS(SecurityFramework):
    """The proposed ADAPT-MAS framework."""

    def __init__(self, agents: List[str]):
        super().__init__(agents)
        self.scores = {agent: {'default': TRUST_INITIAL} for agent in agents}
        self.agents = agents
        self.social_graph = nx.DiGraph()
        self.social_graph.add_nodes_from(agents)
        self.performance_history = {agent: [] for agent in agents}

    def get_initial_score(self) -> float:
        return TRUST_INITIAL

    def get_trust_score(self, agent_id: str, context: str) -> float:
        """Gets the trust score for a specific context, creating it if it doesn't exist."""
        if not isinstance(self.scores.get(agent_id), dict):
            self.scores[agent_id] = {'default': TRUST_INITIAL}
        return self.scores[agent_id].setdefault(context, TRUST_INITIAL)

    def update_scores(self,
                      task_category: str,
                      peer_reviews: Dict[str, Dict[str, float]],
                      contribution_scores: Dict[str, float],
                      ground_truth_reward: float = None):
        """Core update function, orchestrating the three main modules."""
        context = task_category

        # --- 1. Dynamic Trust Model: Time Decay ---
        for agent in self.agents:
            current_trust = self.get_trust_score(agent, context)
            self.scores[agent][context] = current_trust * TRUST_TIME_DECAY_FACTOR

        # --- 2. Dynamic Trust Model: New Evidence Fusion ---
        evidence = {}
        if ground_truth_reward is not None:
            evidence = {agent_id: ground_truth_reward * csc for agent_id, csc in contribution_scores.items()}
        elif peer_reviews:
            evidence = self._calculate_cis(peer_reviews, context)

        for agent_id, E_t in evidence.items():
            if agent_id in self.scores:
                current_trust = self.get_trust_score(agent_id, context)
                new_trust = current_trust + TRUST_LEARNING_RATE * E_t
                self.scores[agent_id][context] = new_trust
                self.performance_history[agent_id].append(E_t)

        # --- 3. Social Graph Analysis ---
        if peer_reviews:
            self._update_social_graph(peer_reviews)
            colluding_groups = self._detect_collusion()

            # --- MODIFICATION START ---
            # Only apply collusion penalty IF the round was NOT successful.
            # This prevents penalizing agents for good collaboration that leads to a correct answer.
            # For subjective tasks (ground_truth_reward is None), the penalty is always active.
            apply_penalty = ground_truth_reward is None or ground_truth_reward < 0

            if colluding_groups and apply_penalty:
                print(f"ADAPT-MAS: Collusion detected in a FAILED round, applying penalty. Groups: {colluding_groups}")
                for group in colluding_groups:
                    for agent_id in group:
                        if agent_id in self.scores:
                            current_trust = self.get_trust_score(agent_id, context)
                            self.scores[agent_id][context] = current_trust * TRUST_PENALTY_FACTOR
            elif colluding_groups and not apply_penalty:
                print(
                    f"ADAPT-MAS: Collusion detected but round was SUCCESSFUL. Penalty has been waived. Groups: {colluding_groups}")
            # --- MODIFICATION END ---

        # --- Normalize all scores to the [0, 1] range ---
        for agent in self.agents:
            score = self.get_trust_score(agent, context)
            self.scores[agent][context] = max(0.0, min(1.0, score))

        current_scores_snapshot = {agent: self.get_trust_score(agent, context) for agent in self.agents}
        self.history.append(current_scores_snapshot)
        print(f"ADAPT-MAS Updated (Context: {context}): {current_scores_snapshot}")

    def _calculate_cis(self, peer_reviews: Dict[str, Dict[str, float]], context: str) -> Dict[str, float]:
        """Calculates the Contribution Influence Score (CIS)."""
        cis_scores = {agent: 0.0 for agent in self.agents}
        total_trust_weight = {agent: 1e-9 for agent in self.agents}

        for reviewer_id, reviews in peer_reviews.items():
            reviewer_trust = self.get_trust_score(reviewer_id, context)
            for reviewee_id, score in reviews.items():
                if reviewee_id in cis_scores:
                    cis_scores[reviewee_id] += score * reviewer_trust
                    total_trust_weight[reviewee_id] += reviewer_trust

        for agent_id in cis_scores:
            if total_trust_weight[agent_id] > 1e-9:
                cis_scores[agent_id] /= total_trust_weight[agent_id]
        return cis_scores

    def _update_social_graph(self, peer_reviews: Dict[str, Dict[str, float]]):
        """Update social graph edge weights based on peer reviews."""
        if not peer_reviews: return
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
        """Detects colluding groups using community detection algorithms."""
        strong_links_graph = nx.Graph()
        for u, v, data in self.social_graph.edges(data=True):
            if data.get('weight', 0) > GRAPH_COLLUSION_THRESHOLD:
                strong_links_graph.add_edge(u, v, weight=data['weight'])

        if not strong_links_graph.nodes() or not strong_links_graph.edges():
            return []

        communities_generator = community.louvain_communities(strong_links_graph, weight='weight')
        potential_groups = [list(c) for c in communities_generator if len(c) > 1]
        colluding_groups = []
        for group in potential_groups:
            if self._calculate_suspicion_score(group) > COMMUNITY_SUSPICION_THRESHOLD:
                colluding_groups.append(group)
        return colluding_groups

    def _calculate_suspicion_score(self, group: List[str]) -> float:
        """Calculates the suspicion score for a community."""
        if len(group) < 2: return 0.0

        group_set = set(group)
        internal_edges, external_edges = 0, 0
        internal_weight_sum, external_weight_sum = 0.0, 0.0
        internal_reviews, external_reviews = [], []

        for u in group:
            if u not in self.social_graph: continue
            for v, data in self.social_graph.adj[u].items():
                weight = data.get('weight', 0.5)
                score = weight * 2 - 1
                if v in group_set:
                    internal_edges += 1
                    internal_weight_sum += weight
                    internal_reviews.append(score)
                else:
                    external_edges += 1
                    external_weight_sum += weight
                    external_reviews.append(score)

        internal_cohesion = (internal_weight_sum / internal_edges) if internal_edges > 0 else 0
        external_isolation = 1.0 - (external_weight_sum / external_edges if external_edges > 0 else 0)
        avg_internal_review = np.mean(internal_reviews) if internal_reviews else 0.5
        avg_external_review = np.mean(external_reviews) if external_reviews else 0.5
        evaluation_bias = max(0, avg_internal_review - avg_external_review)

        suspicion_score = (internal_cohesion * 0.4 + external_isolation * 0.3 + evaluation_bias * 0.3)
        return suspicion_score
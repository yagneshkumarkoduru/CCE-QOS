from __future__ import annotations

import math
import random
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Set, Tuple

from core_types import OperatorGraph


@dataclass(slots=True)
class ScheduleResult:
    strategy: str
    order: List[int]
    score: float
    metadata: Dict


class SchedulingEngine:
    """
    Classical schedule search engine over topologically valid orders.
    """

    def __init__(self, graph: OperatorGraph, random_seed: int = 7) -> None:
        self.graph = graph
        self.rng = random.Random(random_seed)
        self.levels = graph.compute_levels()
        self.critical_path = graph.critical_path_cycles()
        self.descendants = graph.descendant_count()
        self.max_critical = max(self.critical_path.values()) if self.critical_path else 1.0

    def _priority_score(self, node_id: int, penalties: Dict[str, float], frontier_width: int) -> float:
        node = self.graph.node_by_id[node_id]

        cp_norm = self.critical_path[node_id] / max(1.0, self.max_critical)
        depth = self.levels[node_id]
        fanout = len(self.graph.children_by_id[node_id])
        desc = self.descendants[node_id]

        critical_bonus = float(node.attrs.get("criticality", 1.0))
        mem_pressure = (node.input_bytes + node.output_bytes) * penalties.get("dram_pressure", 1.0)
        sram_hint = float(node.attrs.get("sram_hint", node.output_bytes * 0.75))
        sram_risk = max(0.0, node.output_bytes - sram_hint) * penalties.get("sram_capacity", 1.0)
        bandwidth_risk = (node.input_bytes / max(1.0, node.compute_cycles)) * penalties.get("bandwidth_capacity", 1.0)

        frontier_bonus = 1.0 / max(1.0, frontier_width)
        unlock_gain = 0.9 * fanout + 0.08 * desc

        return (
            1.6 * cp_norm * node.compute_cycles * critical_bonus
            + 0.18 * mem_pressure
            + 0.11 * sram_risk
            + 0.15 * bandwidth_risk
            + 0.03 * depth
            - unlock_gain
            - 0.75 * frontier_bonus
        )

    def _biased_complete_order(
        self,
        prefix: Sequence[int],
        penalties: Dict[str, float],
        exploration_noise: float,
    ) -> List[int]:
        scheduled = set(int(x) for x in prefix)
        order = list(int(x) for x in prefix)

        while len(order) < len(self.graph.nodes):
            ready = self.graph.ready_nodes(scheduled)
            if not ready:
                break

            frontier_width = len(ready)
            scores = []
            for node_id in ready:
                base = self._priority_score(node_id, penalties, frontier_width)
                noise = self.rng.gauss(0.0, exploration_noise) if exploration_noise > 0.0 else 0.0
                scores.append((base + noise, node_id))
            scores.sort(key=lambda item: item[0])
            chosen = scores[0][1]
            order.append(chosen)
            scheduled.add(chosen)
        return order

    def _random_topological_order(self) -> List[int]:
        scheduled = set()
        order: List[int] = []
        while len(order) < len(self.graph.nodes):
            ready = self.graph.ready_nodes(scheduled)
            if not ready:
                break
            choice = self.rng.choice(ready)
            order.append(choice)
            scheduled.add(choice)
        return order

    def _rollout_estimate(
        self,
        prefix: Sequence[int],
        penalties: Dict[str, float],
        evaluator: Callable[[Sequence[int]], float],
        trials: int,
        exploration_noise: float,
    ) -> float:
        scores = []
        for _ in range(max(1, trials)):
            order = self._biased_complete_order(prefix, penalties, exploration_noise)
            if len(order) != len(self.graph.nodes):
                continue
            scores.append(evaluator(order))

        if not scores:
            return float("inf")
        if len(scores) == 1:
            return scores[0]
        return statistics.fmean(scores) + 0.35 * statistics.pstdev(scores)

    def greedy(self, penalties: Dict[str, float]) -> ScheduleResult:
        order = self._biased_complete_order(prefix=[], penalties=penalties, exploration_noise=0.0)
        return ScheduleResult(strategy="Greedy", order=order, score=0.0, metadata={"penalties": penalties})

    def lookahead(
        self,
        penalties: Dict[str, float],
        lookahead_depth: int,
        evaluator: Callable[[Sequence[int]], float],
    ) -> ScheduleResult:
        rollout_trials = 4
        branch_factor = 3
        cached_scores: Dict[Tuple[int, ...], float] = {}

        def recursive_score(prefix: List[int], scheduled_now: Set[int], depth: int) -> float:
            key = tuple(prefix)
            if key in cached_scores and depth <= 1:
                return cached_scores[key]

            if depth == 0 or len(prefix) == len(self.graph.nodes):
                score = self._rollout_estimate(
                    prefix=prefix,
                    penalties=penalties,
                    evaluator=evaluator,
                    trials=rollout_trials,
                    exploration_noise=0.3,
                )
                cached_scores[key] = score
                return score

            ready = self.graph.ready_nodes(scheduled_now)
            if not ready:
                return float("inf")

            ranked_ready = sorted(ready, key=lambda n: self._priority_score(n, penalties, len(ready)))[:branch_factor]
            local_scores = []
            for node_id in ranked_ready:
                next_prefix = list(prefix)
                next_prefix.append(node_id)
                next_scheduled = set(scheduled_now)
                next_scheduled.add(node_id)
                local_scores.append(recursive_score(next_prefix, next_scheduled, depth - 1))

            score = min(local_scores) if local_scores else float("inf")
            cached_scores[key] = score
            return score

        scheduled = set()
        order: List[int] = []
        while len(order) < len(self.graph.nodes):
            ready = self.graph.ready_nodes(scheduled)
            if not ready:
                break

            ranked_ready = sorted(ready, key=lambda n: self._priority_score(n, penalties, len(ready)))[:branch_factor]
            candidates = []
            for node_id in ranked_ready:
                prefix = list(order)
                prefix.append(node_id)
                scheduled_probe = set(scheduled)
                scheduled_probe.add(node_id)
                score = recursive_score(prefix, scheduled_probe, lookahead_depth)
                candidates.append((score, node_id))

            candidates.sort(key=lambda item: item[0])
            selected = candidates[0][1]
            order.append(selected)
            scheduled.add(selected)

        completed = self._biased_complete_order(order, penalties, exploration_noise=0.0)
        return ScheduleResult(
            strategy="Lookahead",
            order=completed,
            score=evaluator(completed),
            metadata={
                "depth": lookahead_depth,
                "rollout_trials": rollout_trials,
                "branch_factor": branch_factor,
                "penalties": penalties,
            },
        )

    def beam_search(
        self,
        penalties: Dict[str, float],
        beam_width: int,
        evaluator: Callable[[Sequence[int]], float],
    ) -> ScheduleResult:
        expansion_topk = 4
        beam: List[Tuple[List[int], Set[int], float]] = [([], set(), 0.0)]

        for _ in range(len(self.graph.nodes)):
            expanded: List[Tuple[List[int], Set[int], float]] = []
            for prefix, scheduled, _ in beam:
                ready = self.graph.ready_nodes(scheduled)
                if not ready:
                    continue

                ranked_ready = sorted(ready, key=lambda n: self._priority_score(n, penalties, len(ready)))[:expansion_topk]
                for node_id in ranked_ready:
                    next_prefix = list(prefix)
                    next_prefix.append(node_id)
                    next_scheduled = set(scheduled)
                    next_scheduled.add(node_id)

                    score = self._rollout_estimate(
                        prefix=next_prefix,
                        penalties=penalties,
                        evaluator=evaluator,
                        trials=3,
                        exploration_noise=0.25,
                    )
                    expanded.append((next_prefix, next_scheduled, score))

            if not expanded:
                break

            expanded.sort(key=lambda item: item[2])
            next_beam: List[Tuple[List[int], Set[int], float]] = []
            signatures: Set[Tuple[int, ...]] = set()
            for candidate in expanded:
                prefix = candidate[0]
                signature = tuple(prefix[-3:]) if len(prefix) >= 3 else tuple(prefix)
                if signature in signatures:
                    continue
                signatures.add(signature)
                next_beam.append(candidate)
                if len(next_beam) >= beam_width:
                    break
            beam = next_beam if next_beam else expanded[:beam_width]

        best_prefix = beam[0][0] if beam else []
        best_order = self._biased_complete_order(best_prefix, penalties, exploration_noise=0.0)
        return ScheduleResult(
            strategy="Beam Search",
            order=best_order,
            score=evaluator(best_order),
            metadata={
                "beam_width": beam_width,
                "expansion_topk": expansion_topk,
                "penalties": penalties,
            },
        )

    def _is_valid(self, order: Sequence[int]) -> bool:
        return self.graph.is_valid_order(order)

    def _neighbor(self, order: Sequence[int], max_tries: int = 60) -> List[int]:
        base = list(order)
        n = len(base)
        if n < 2:
            return base

        for _ in range(max_tries):
            move = self.rng.choice(["swap", "insert", "block"])
            proposal = list(base)

            if move == "swap":
                i = self.rng.randrange(0, n)
                j = self.rng.randrange(0, n)
                if i == j:
                    continue
                proposal[i], proposal[j] = proposal[j], proposal[i]
            elif move == "insert":
                i = self.rng.randrange(0, n)
                j = self.rng.randrange(0, n)
                if i == j:
                    continue
                value = proposal.pop(i)
                proposal.insert(j, value)
            else:
                if n < 4:
                    continue
                left = self.rng.randrange(0, n - 2)
                right = self.rng.randrange(left + 1, min(n, left + 4))
                proposal[left:right] = reversed(proposal[left:right])

            if self._is_valid(proposal):
                return proposal
        return base

    def simulated_annealing(
        self,
        penalties: Dict[str, float],
        evaluator: Callable[[Sequence[int]], float],
        iterations: int,
        start_temp: float,
        end_temp: float,
        initial_order: Sequence[int] | None = None,
    ) -> ScheduleResult:
        if initial_order is not None and self._is_valid(initial_order):
            current = list(int(x) for x in initial_order)
        else:
            current = self._biased_complete_order([], penalties, exploration_noise=0.0)
            if not self._is_valid(current):
                current = self._random_topological_order()
        current_score = evaluator(current)

        best = list(current)
        best_score = current_score
        accepted = 0
        stagnant_steps = 0
        tabu_limit = 80
        tabu = deque()
        tabu_set: Set[Tuple[int, ...]] = set()

        for step in range(1, iterations + 1):
            t_ratio = step / max(1, iterations)
            base_temp = start_temp * ((end_temp / start_temp) ** t_ratio)
            temp = base_temp * 1.6 if stagnant_steps > 30 else base_temp

            candidate = self._neighbor(current)
            signature = tuple(candidate)
            if signature in tabu_set:
                continue

            candidate_score = evaluator(candidate)
            delta = candidate_score - current_score
            accept = delta <= 0.0 or self.rng.random() < math.exp(-delta / max(temp, 1e-6))

            if accept:
                current = candidate
                current_score = candidate_score
                accepted += 1

                if current_score + 1e-9 < best_score:
                    best = list(current)
                    best_score = current_score
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1

                tabu.append(signature)
                tabu_set.add(signature)
                if len(tabu) > tabu_limit:
                    old = tabu.popleft()
                    tabu_set.discard(old)
            else:
                stagnant_steps += 1

        return ScheduleResult(
            strategy="Simulated Annealing",
            order=best,
            score=best_score,
            metadata={
                "iterations": iterations,
                "start_temp": start_temp,
                "end_temp": end_temp,
                "acceptance_ratio": accepted / max(1, iterations),
                "tabu_size": tabu_limit,
                "warm_started": initial_order is not None,
                "penalties": penalties,
            },
        )

    def feasibility_preserving_descent(
        self,
        initial_order: Sequence[int],
        score_fn: Callable[[Sequence[int]], Tuple[float, float]],
        iterations: int = 200,
        tolerance: float = 1e-9,
    ) -> Tuple[List[int], float, float]:
        """
        Feasibility-preserving local descent on the true objective.

        score_fn(order) must return (objective, constraint_mass). Starting from
        a schedule, only neighbor moves are accepted that either strictly
        reduce the constraint mass or keep it unchanged while strictly
        reducing the objective; the constraint mass therefore never increases
        and the objective improves monotonically at equal constraint mass.
        Uses this engine's swap/insert/block-reverse move operators.
        """
        current = list(int(x) for x in initial_order)
        if not self._is_valid(current):
            raise ValueError("feasibility_preserving_descent requires a valid starting order.")

        current_objective, current_constraint = score_fn(current)
        for _ in range(max(0, iterations)):
            proposal = self._neighbor(current)
            proposal_objective, proposal_constraint = score_fn(proposal)
            improved = (
                proposal_constraint < current_constraint - 1e-12
                or (
                    proposal_constraint <= current_constraint + 1e-12
                    and proposal_objective < current_objective - tolerance
                )
            )
            if improved:
                current = proposal
                current_objective = proposal_objective
                current_constraint = proposal_constraint
        return current, current_objective, current_constraint

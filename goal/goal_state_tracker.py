"""
State tracking for goal satisfaction and constraint monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np

from .goal_schema import GoalDefinition


@dataclass
class GoalSatisfactionSnapshot:
    """
    Aggregated view of how well the current training run satisfies the goal.
    """

    cumulative_return: float
    annualized_volatility: float
    max_drawdown: float
    latency_ms: float
    capital_in_use: float
    utility_score: float
    constraint_violations: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        payload = {
            "cumulative_return": self.cumulative_return,
            "annualized_volatility": self.annualized_volatility,
            "max_drawdown": self.max_drawdown,
            "latency_ms": self.latency_ms,
            "capital_in_use": self.capital_in_use,
            "utility_score": self.utility_score,
        }
        if self.constraint_violations:
            payload["constraint_violations"] = dict(self.constraint_violations)
        return payload


@dataclass
class GoalHealthSnapshot:
    utility_history: list
    negative_steps: int
    last_utility: float
    average_utility: float
    violation_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utility_history": self.utility_history,
            "negative_steps": self.negative_steps,
            "last_utility": self.last_utility,
            "average_utility": self.average_utility,
            "violation_counts": dict(self.violation_counts),
        }


class GoalStateTracker:
    """
    Tracks running metrics for a training/evaluation session and
    computes goal satisfaction snapshots on demand.
    """

    def __init__(self, goal: GoalDefinition, periods_per_year: int = 252):
        self.goal = goal
        self.periods_per_year = periods_per_year

        self._returns = []
        self._latencies = []
        self._capital_usage = []
        self._max_capital = goal.constraints.capital_limit or 0.0
        self._utility_history = []
        self._current_drawdown = 0.0
        self._peak = 1.0
        self._wealth = 1.0
        self._violation_counts: Dict[str, int] = {}
        self._negative_utility_steps = 0

    def reset(self):
        self._returns.clear()
        self._latencies.clear()
        self._capital_usage.clear()
        self._utility_history.clear()
        self._current_drawdown = 0.0
        self._peak = 1.0
        self._wealth = 1.0
        self._violation_counts.clear()
        self._negative_utility_steps = 0

    def update_step(
        self,
        realized_return: float,
        latency_ms: Optional[float] = None,
        capital_in_use: Optional[float] = None,
    ):
        self._returns.append(float(realized_return))
        self._wealth *= 1.0 + float(realized_return)
        self._peak = max(self._peak, self._wealth)
        self._current_drawdown = min(
            self._current_drawdown, (self._wealth - self._peak) / self._peak
        )

        if latency_ms is not None:
            self._latencies.append(float(latency_ms))

        if capital_in_use is not None:
            self._capital_usage.append(float(capital_in_use))
            self._max_capital = max(self._max_capital, float(capital_in_use))

        snapshot = self.snapshot()
        self._utility_history.append(snapshot.utility_score)
        if snapshot.utility_score < 0:
            self._negative_utility_steps += 1

        if snapshot.constraint_violations:
            for key in snapshot.constraint_violations.keys():
                self._violation_counts[key] = self._violation_counts.get(key, 0) + 1

    def snapshot(self) -> GoalSatisfactionSnapshot:
        returns = np.array(self._returns, dtype=np.float64) if self._returns else np.array([0.0])
        cumulative_return = float(np.prod(1.0 + returns) - 1.0)

        if returns.size > 1:
            volatility = float(np.std(returns) * np.sqrt(self.periods_per_year))
        else:
            volatility = 0.0

        max_drawdown = float(self._current_drawdown)
        latency_ms = float(np.mean(self._latencies)) if self._latencies else 0.0
        capital_in_use = float(np.mean(self._capital_usage)) if self._capital_usage else 0.0

        # Utility score (normalized weighted preferences)
        weights = self.goal.normalized_weights
        # Interpret risk as volatility magnitude, latency as mean latency, capital efficiency as inverse of capital usage
        utility_score = (
            weights["return_weight"] * cumulative_return
            - weights["risk_weight"] * volatility
            - weights["latency_weight"] * latency_ms
            + weights["capital_efficiency_weight"] * self._capital_efficiency(capital_in_use)
            + weights["style_alignment_weight"] * self._style_alignment_bonus()
        )

        constraint_violations = self._constraint_violations(
            volatility=volatility,
            drawdown=max_drawdown,
            capital=capital_in_use,
            latency=latency_ms,
        )

        return GoalSatisfactionSnapshot(
            cumulative_return=cumulative_return,
            annualized_volatility=volatility,
            max_drawdown=max_drawdown,
            latency_ms=latency_ms,
            capital_in_use=capital_in_use,
            utility_score=float(utility_score),
            constraint_violations=constraint_violations,
        )

    def health_snapshot(self) -> GoalHealthSnapshot:
        average_utility = float(np.mean(self._utility_history)) if self._utility_history else 0.0
        last_utility = self._utility_history[-1] if self._utility_history else 0.0
        return GoalHealthSnapshot(
            utility_history=list(self._utility_history),
            negative_steps=self._negative_utility_steps,
            last_utility=float(last_utility),
            average_utility=average_utility,
            violation_counts=dict(self._violation_counts),
        )

    def _capital_efficiency(self, capital_in_use: float) -> float:
        if capital_in_use <= 0:
            return 0.0

        limit = self.goal.constraints.capital_limit
        if limit is None or limit <= 0:
            # Higher capital usage is interpreted as lower efficiency, so take inverse
            return 1.0 / (1.0 + capital_in_use)

        utilization = capital_in_use / limit
        # Efficiency bonus for staying within limit, penalty for exceeding
        if utilization <= 1.0:
            return 1.0 - utilization
        return -(utilization - 1.0)

    def _style_alignment_bonus(self) -> float:
        """
        Placeholder style alignment term until agents provide feedback.
        Returns small positive bump for conservative preferences when drawdown is low,
        or aggressive preferences when returns are high.
        """
        if not self._returns:
            return 0.0

        avg_return = float(np.mean(self._returns))
        max_drawdown = float(self._current_drawdown)
        appetite = self.goal.style.risk_appetite.lower()

        if appetite == "conservative":
            return max(0.0, -max_drawdown)
        if appetite == "aggressive":
            return max(0.0, avg_return)
        return 0.5 * max(0.0, avg_return) - 0.5 * max(0.0, -max_drawdown)

    def _constraint_violations(
        self,
        volatility: float,
        drawdown: float,
        capital: float,
        latency: float,
    ) -> Dict[str, float]:
        violations: Dict[str, float] = {}
        constraints = self.goal.constraints

        if constraints.max_var_95 is not None:
            # Approximate VaR via volatility assuming Gaussian returns
            approx_var = 1.65 * volatility
            if approx_var > constraints.max_var_95:
                violations["max_var_95"] = approx_var - constraints.max_var_95

        if constraints.max_cvar_95 is not None:
            approx_cvar = 2.06 * volatility
            if approx_cvar > constraints.max_cvar_95:
                violations["max_cvar_95"] = approx_cvar - constraints.max_cvar_95

        if constraints.max_drawdown is not None:
            if abs(drawdown) > abs(constraints.max_drawdown):
                violations["max_drawdown"] = abs(drawdown) - abs(constraints.max_drawdown)

        if constraints.capital_limit is not None and capital > constraints.capital_limit:
            violations["capital_limit"] = capital - constraints.capital_limit

        if constraints.latency_limit_ms is not None and latency > constraints.latency_limit_ms:
            violations["latency_limit_ms"] = latency - constraints.latency_limit_ms

        if violations:
            return violations
        return {}


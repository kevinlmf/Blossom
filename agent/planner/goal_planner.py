"""
Goal-driven planner that turns personalized objectives into system directives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from goal import GoalDefinition, GoalSatisfactionSnapshot


@dataclass
class PlannerInputs:
    market_regime: str
    market_conditions: Dict[str, float]
    baseline_metrics: Dict[str, float]
    goal_snapshot: Optional[GoalSatisfactionSnapshot] = None


@dataclass
class PlannerDirective:
    allocation_targets: Dict[str, float]
    allocation_floor: float
    risk_targets: Dict[str, float]
    latency_budget_ms: Optional[float]
    capital_budget: Optional[float]
    constraint_overrides: Dict[str, float] = field(default_factory=dict)
    style_notes: str = ""

    def to_dict(self) -> Dict[str, float]:
        return {
            "allocation_targets": dict(self.allocation_targets),
            "allocation_floor": self.allocation_floor,
            "risk_targets": dict(self.risk_targets),
            "latency_budget_ms": self.latency_budget_ms,
            "capital_budget": self.capital_budget,
            "constraint_overrides": dict(self.constraint_overrides),
            "style_notes": self.style_notes,
        }


class GoalPlanner:
    """
    Heuristic goal planner that maps a GoalDefinition into actionable targets
    for the allocator, agents, and controllers.
    """

    def __init__(self, goal: GoalDefinition):
        self.goal = goal

    def generate_directive(self, inputs: PlannerInputs) -> PlannerDirective:
        goal_weights = self.goal.normalized_weights
        style = self.goal.style
        constraints = self.goal.constraints

        # Base allocation preference across frequency bands
        allocation = self._allocation_from_preferences(goal_weights, style)

        # Risk targets adapt to goal emphasis and market regime
        risk_targets = self._risk_targets(goal_weights, style, inputs.market_regime)

        # Minimum allocation floor ensures diversification
        allocation_floor = self._allocation_floor(style)

        latency_budget = constraints.latency_limit_ms
        capital_budget = constraints.capital_limit

        constraint_overrides = {}
        if constraints.max_drawdown is not None:
            constraint_overrides["max_drawdown"] = constraints.max_drawdown
        if constraints.max_var_95 is not None:
            constraint_overrides["max_var_95"] = constraints.max_var_95
        if constraints.max_cvar_95 is not None:
            constraint_overrides["max_cvar_95"] = constraints.max_cvar_95

        style_notes = self._style_notes(style, inputs.market_regime)

        # Adjust allocations dynamically based on goal satisfaction feedback
        if inputs.goal_snapshot:
            allocation = self._adjust_allocation_with_feedback(
                allocation, goal_weights, inputs.goal_snapshot
            )

        directive = PlannerDirective(
            allocation_targets=allocation,
            allocation_floor=allocation_floor,
            risk_targets=risk_targets,
            latency_budget_ms=latency_budget,
            capital_budget=capital_budget,
            constraint_overrides=constraint_overrides,
            style_notes=style_notes,
        )

        return directive

    def _allocation_from_preferences(
        self, goal_weights: Dict[str, float], style
    ) -> Dict[str, float]:
        return_weight = goal_weights["return_weight"]
        risk_weight = goal_weights["risk_weight"]

        base_hft = 0.3 + 0.2 * return_weight
        base_mft = 0.4 + 0.1 * (return_weight - risk_weight)
        base_lft = 0.3 + 0.2 * risk_weight

        # Style adjustments
        execution = style.execution_style.lower()
        if execution == "aggressive":
            base_hft += 0.1
            base_lft -= 0.1
        elif execution == "conservative":
            base_lft += 0.1
            base_hft -= 0.1

        if style.risk_appetite.lower() == "conservative":
            base_lft += 0.05
            base_hft -= 0.05
        elif style.risk_appetite.lower() == "aggressive":
            base_hft += 0.05
            base_lft -= 0.05

        allocations = {
            "hft": max(base_hft, 0.05),
            "mft": max(base_mft, 0.05),
            "lft": max(base_lft, 0.05),
        }

        total = sum(allocations.values())
        if total == 0:
            return {"hft": 1 / 3, "mft": 1 / 3, "lft": 1 / 3}
        return {k: v / total for k, v in allocations.items()}

    def _risk_targets(
        self, goal_weights: Dict[str, float], style, market_regime: str
    ) -> Dict[str, float]:
        risk_bias = goal_weights["risk_weight"]
        base_risk = max(0.1, min(0.9, 0.5 + (0.5 - risk_bias)))

        if style.risk_appetite.lower() == "conservative":
            base_risk *= 0.7
        elif style.risk_appetite.lower() == "aggressive":
            base_risk *= 1.2

        regime_modifier = {
            "high_risk": 0.8,
            "high_return": 1.1,
            "stable": 1.0,
        }.get(market_regime, 1.0)

        adjusted_risk = max(0.05, min(1.5, base_risk * regime_modifier))

        return {
            "risk_aversion": adjusted_risk,
            "cvar_budget": goal_weights["risk_weight"],
        }

    def _allocation_floor(self, style) -> float:
        if style.execution_style.lower() == "conservative":
            return 0.08
        if style.execution_style.lower() == "aggressive":
            return 0.03
        return 0.05

    def _style_notes(self, style, market_regime: str) -> str:
        notes = [
            f"Execution style: {style.execution_style}",
            f"Risk appetite: {style.risk_appetite}",
            f"Market regime: {market_regime}",
        ]
        if style.thematic_focus:
            notes.append(f"Thematic focus: {style.thematic_focus}")
        if style.preferred_instruments:
            notes.append(f"Preferred: {', '.join(style.preferred_instruments)}")
        if style.disallowed_instruments:
            notes.append(f"Disallowed: {', '.join(style.disallowed_instruments)}")
        return " | ".join(notes)

    def _adjust_allocation_with_feedback(
        self,
        allocations: Dict[str, float],
        goal_weights: Dict[str, float],
        snapshot: GoalSatisfactionSnapshot,
    ) -> Dict[str, float]:
        adjusted = dict(allocations)
        # If risk violations are present, shift toward lower frequency agents
        if snapshot.constraint_violations:
            penalty = min(0.1, 0.05 + goal_weights["risk_weight"] * 0.1)
            adjusted["lft"] = min(1.0, adjusted["lft"] + penalty)
            adjusted["hft"] = max(0.05, adjusted["hft"] - penalty / 2)
            adjusted["mft"] = max(0.05, adjusted["mft"] - penalty / 2)

        # If utility is high and return weight dominates, reward HFT/MFT
        if snapshot.utility_score > 0 and goal_weights["return_weight"] > 0.4:
            bonus = min(0.1, snapshot.utility_score * 0.1)
            adjusted["hft"] = min(1.0, adjusted["hft"] + bonus * 0.6)
            adjusted["mft"] = min(1.0, adjusted["mft"] + bonus * 0.4)
            adjusted["lft"] = max(0.05, adjusted["lft"] - bonus)

        total = sum(adjusted.values())
        if total <= 0:
            return allocations
        return {k: max(0.0, v / total) for k, v in adjusted.items()}


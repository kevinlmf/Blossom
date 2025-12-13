"""
Goal definition schemas for the goal-conditioned trading system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UtilityWeights:
    """
    Weights that shape the personalized utility function:

        U = w_return * Return
            - w_risk * Risk
            - w_latency * Latency
            + w_capital_efficiency * CapitalEfficiency
            + w_style * StyleAlignment

    All weights are normalized to sum to 1.0 when requested through
    `normalized()`.
    """

    return_weight: float = 0.4
    risk_weight: float = 0.3
    latency_weight: float = 0.1
    capital_efficiency_weight: float = 0.1
    style_alignment_weight: float = 0.1

    def normalized(self) -> Dict[str, float]:
        weights = {
            "return_weight": max(self.return_weight, 0.0),
            "risk_weight": max(self.risk_weight, 0.0),
            "latency_weight": max(self.latency_weight, 0.0),
            "capital_efficiency_weight": max(self.capital_efficiency_weight, 0.0),
            "style_alignment_weight": max(self.style_alignment_weight, 0.0),
        }
        total = sum(weights.values())
        if total <= 0:
            # Fallback to equal weights if all zero/non-positive
            n = len(weights)
            return {key: 1.0 / n for key in weights}
        return {key: value / total for key, value in weights.items()}


@dataclass
class GoalConstraints:
    """
    Hard and soft constraints that the planner and controllers must satisfy.
    """

    max_var_95: Optional[float] = None
    max_cvar_95: Optional[float] = None
    max_drawdown: Optional[float] = None
    capital_limit: Optional[float] = None
    latency_limit_ms: Optional[float] = None
    turnover_limit: Optional[float] = None
    leverage_limit: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class GoalStylePreferences:
    """
    Qualitative preferences used by the planner to bias allocations/strategies.
    """

    execution_style: str = "balanced"  # e.g. conservative, aggressive, balanced
    risk_appetite: str = "balanced"  # e.g. conservative, moderate, aggressive
    preferred_instruments: List[str] = field(default_factory=list)
    disallowed_instruments: List[str] = field(default_factory=list)
    thematic_focus: Optional[str] = None  # e.g. ESG, tech, defensive
    comments: Optional[str] = None


@dataclass
class GoalDefinition:
    """
    Full specification of a personalized trading goal.
    """

    name: str = "default_goal"
    owner: str = "anonymous"
    description: str = "Balanced risk-adjusted wealth growth."
    utility_weights: UtilityWeights = field(default_factory=UtilityWeights)
    constraints: GoalConstraints = field(default_factory=GoalConstraints)
    style: GoalStylePreferences = field(default_factory=GoalStylePreferences)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "owner": self.owner,
            "description": self.description,
            "utility_weights": self.utility_weights.normalized(),
            "constraints": {
                "max_var_95": self.constraints.max_var_95,
                "max_cvar_95": self.constraints.max_cvar_95,
                "max_drawdown": self.constraints.max_drawdown,
                "capital_limit": self.constraints.capital_limit,
                "latency_limit_ms": self.constraints.latency_limit_ms,
                "turnover_limit": self.constraints.turnover_limit,
                "leverage_limit": self.constraints.leverage_limit,
                "notes": self.constraints.notes,
            },
            "style": {
                "execution_style": self.style.execution_style,
                "risk_appetite": self.style.risk_appetite,
                "preferred_instruments": list(self.style.preferred_instruments),
                "disallowed_instruments": list(self.style.disallowed_instruments),
                "thematic_focus": self.style.thematic_focus,
                "comments": self.style.comments,
            },
            "metadata": dict(self.metadata),
        }

    @property
    def normalized_weights(self) -> Dict[str, float]:
        return self.utility_weights.normalized()


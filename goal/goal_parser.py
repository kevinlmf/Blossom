"""
Utilities for loading goal definitions from configuration files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .goal_schema import (
    GoalConstraints,
    GoalDefinition,
    GoalStylePreferences,
    UtilityWeights,
)


def _ensure_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Goal configuration file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Goal configuration path is not a file: {path}")
    return path


class GoalParser:
    """
    Parses goal configuration files (YAML/JSON) into GoalDefinition objects.
    """

    SUPPORTED_EXTENSIONS = {".yaml", ".yml", ".json"}

    def __init__(self, default_goal: Optional[GoalDefinition] = None):
        self._default_goal = default_goal or GoalDefinition()

    def load(self, config: Optional[str] = None) -> GoalDefinition:
        if config is None:
            return self._default_goal

        path = Path(config).expanduser().resolve()
        path = _ensure_path(path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported goal config extension '{path.suffix}'. "
                f"Supported: {supported}"
            )

        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        else:
            with path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)

        if raw is None:
            raise ValueError(f"Goal configuration file {path} is empty.")

        return self.from_dict(raw)

    def from_dict(self, payload: Dict[str, Any]) -> GoalDefinition:
        """
        Build GoalDefinition from a dictionary payload.
        Missing sections fall back to defaults from the base goal.
        """
        base = self._default_goal

        utility_payload = payload.get("utility_weights", {})
        constraint_payload = payload.get("constraints", {})
        style_payload = payload.get("style", payload.get("style_preferences", {}))

        utility = UtilityWeights(
            return_weight=utility_payload.get(
                "return_weight", base.utility_weights.return_weight
            ),
            risk_weight=utility_payload.get(
                "risk_weight", base.utility_weights.risk_weight
            ),
            latency_weight=utility_payload.get(
                "latency_weight", base.utility_weights.latency_weight
            ),
            capital_efficiency_weight=utility_payload.get(
                "capital_efficiency_weight",
                base.utility_weights.capital_efficiency_weight,
            ),
            style_alignment_weight=utility_payload.get(
                "style_alignment_weight",
                base.utility_weights.style_alignment_weight,
            ),
        )

        constraints = GoalConstraints(
            max_var_95=constraint_payload.get("max_var_95", base.constraints.max_var_95),
            max_cvar_95=constraint_payload.get(
                "max_cvar_95", base.constraints.max_cvar_95
            ),
            max_drawdown=constraint_payload.get(
                "max_drawdown", base.constraints.max_drawdown
            ),
            capital_limit=constraint_payload.get(
                "capital_limit", base.constraints.capital_limit
            ),
            latency_limit_ms=constraint_payload.get(
                "latency_limit_ms", base.constraints.latency_limit_ms
            ),
            turnover_limit=constraint_payload.get(
                "turnover_limit", base.constraints.turnover_limit
            ),
            leverage_limit=constraint_payload.get(
                "leverage_limit", base.constraints.leverage_limit
            ),
            notes=constraint_payload.get("notes", base.constraints.notes),
        )

        style = GoalStylePreferences(
            execution_style=style_payload.get(
                "execution_style", base.style.execution_style
            ),
            risk_appetite=style_payload.get(
                "risk_appetite", base.style.risk_appetite
            ),
            preferred_instruments=style_payload.get(
                "preferred_instruments", list(base.style.preferred_instruments)
            ),
            disallowed_instruments=style_payload.get(
                "disallowed_instruments", list(base.style.disallowed_instruments)
            ),
            thematic_focus=style_payload.get(
                "thematic_focus", base.style.thematic_focus
            ),
            comments=style_payload.get("comments", base.style.comments),
        )

        definition = GoalDefinition(
            name=payload.get("name", base.name),
            owner=payload.get("owner", base.owner),
            description=payload.get("description", base.description),
            utility_weights=utility,
            constraints=constraints,
            style=style,
            metadata={**base.metadata, **payload.get("metadata", {})},
        )

        return definition


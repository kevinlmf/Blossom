"""
Goal-driven trading system components.

This package provides:
    - Data schemas for personalized trading goals
    - Parsing utilities for loading goal definitions
    - State trackers for monitoring goal satisfaction
"""

from .goal_schema import (
    GoalConstraints,
    GoalDefinition,
    GoalStylePreferences,
    UtilityWeights,
)
from .goal_parser import GoalParser
from .goal_state_tracker import GoalStateTracker, GoalSatisfactionSnapshot

__all__ = [
    "GoalConstraints",
    "GoalDefinition",
    "GoalStylePreferences",
    "GoalParser",
    "GoalSatisfactionSnapshot",
    "GoalStateTracker",
    "UtilityWeights",
]


from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .memory.strategy_memory_bank import StrategyMemoryBank, StrategyCase


@dataclass
class PretrainResult:
    """Summary of offline pretraining for a single agent."""

    agent_type: str
    regime_type: str
    selected_cases: List[str]
    avg_sharpe: float
    avg_return: float
    avg_drawdown: float
    notes: str

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class OfflinePretrainer:
    """Lightweight Offline RL / Behavior Cloning warm-up stage.

    In the current implementation we do not train a full model offline,
    but we aggregate the top-K memory cases to produce a stable starting
    point and metadata that can be used for reporting or future
    pretraining hooks.
    """

    def __init__(self, memory_bank: StrategyMemoryBank, top_k: int = 3):
        self.memory_bank = memory_bank
        self.top_k = max(1, top_k)

    def run(self, regime_type: str, market_conditions: Dict[str, float]) -> Dict[str, PretrainResult]:
        summary: Dict[str, PretrainResult] = {}
        for agent_type in ['hft', 'mft', 'lft', 'allocator']:
            cases = self.memory_bank.retrieve_best_strategy(
                regime_type=regime_type,
                agent_type=agent_type,
                market_conditions=market_conditions,
                top_k=self.top_k,
            )
            if not cases:
                summary[agent_type] = PretrainResult(
                    agent_type=agent_type,
                    regime_type=regime_type,
                    selected_cases=[],
                    avg_sharpe=0.0,
                    avg_return=0.0,
                    avg_drawdown=0.0,
                    notes="No offline data available; training from scratch.",
                )
                continue

            avg_sharpe = float(sum(case.sharpe_ratio for case in cases) / len(cases))
            avg_return = float(sum(case.total_return for case in cases) / len(cases))
            avg_drawdown = float(sum(case.max_drawdown for case in cases) / len(cases))

            summary[agent_type] = PretrainResult(
                agent_type=agent_type,
                regime_type=regime_type,
                selected_cases=[case.case_id for case in cases],
                avg_sharpe=avg_sharpe,
                avg_return=avg_return,
                avg_drawdown=avg_drawdown,
                notes="Aggregated top memory cases as offline warm-up baseline.",
            )
        return summary



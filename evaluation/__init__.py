"""
Strategy Evaluation Module

Comprehensive evaluation tools for trading strategies including:
- Performance metrics (returns, risk, ratios)
- Risk metrics (VaR, CVaR, drawdowns)
- Trading metrics (win rate, profit factor)
- Visualization tools
- Benchmark comparison
- Progressive learning validation
- Statistical significance testing
"""

from .strategy_evaluator import StrategyEvaluator
from .performance_metrics import PerformanceMetrics, StrategyMetrics
from .visualization import EvaluationVisualizer
from .benchmark_strategies import BenchmarkStrategies, BenchmarkResult, run_all_benchmarks
from .progressive_learning import ProgressiveLearningValidator, LearningSnapshot
from .strategy_comparison import StrategyComparator, ComparisonResult
from .goal_audit import GoalAudit

__all__ = [
    # Core evaluation
    'StrategyEvaluator',
    'PerformanceMetrics',
    'StrategyMetrics',
    'EvaluationVisualizer',
    'GoalAudit',

    # Benchmark strategies
    'BenchmarkStrategies',
    'BenchmarkResult',
    'run_all_benchmarks',

    # Progressive learning validation
    'ProgressiveLearningValidator',
    'LearningSnapshot',

    # Strategy comparison
    'StrategyComparator',
    'ComparisonResult',
]

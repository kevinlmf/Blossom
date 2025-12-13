"""
Experiments Module for Multi-Frequency Trading System

This module provides tools for:
1. Strategy performance monitoring
2. Market regime detection (high risk, high return, stable periods)
3. Real data loading and processing
4. Experiment execution across different market conditions
"""

from .strategy_monitor import create_monitor, StrategyMonitor
from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimePeriod
from .real_data_loader import RealDataLoader
from .experiment_runner import ExperimentRunner

# HMM-based regime detector (optional, requires hmmlearn)
try:
    from .hmm_regime_detector import HMMRegimeDetector, HMMRegimePeriod, compare_methods
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    HMMRegimeDetector = None
    HMMRegimePeriod = None
    compare_methods = None

__all__ = [
    'create_monitor',
    'StrategyMonitor',
    'MarketRegimeDetector',
    'MarketRegime',
    'RegimePeriod',
    'RealDataLoader',
    'ExperimentRunner',
]

if HMM_AVAILABLE:
    __all__.extend(['HMMRegimeDetector', 'HMMRegimePeriod', 'compare_methods'])

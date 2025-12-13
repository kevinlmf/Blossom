"""
Hedging Module

Provides hedging tools to convert excess returns to absolute returns.
"""

from .hedge_manager import HedgeManager, HedgeStrategy, HedgeResult

__all__ = [
    'HedgeManager',
    'HedgeStrategy',
    'HedgeResult'
]





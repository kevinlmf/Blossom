"""
Market Environment Module (Unified Training Ground)

Purpose:
- Provide unified market simulation for all agents
- Simulated and real market data feeds (LOB, OHLCV, macro)
- Reward feedback and transaction constraints
- Acts as the common training ground for HFT/MFT/LFT agents

Environments:
- BaseMarketEnv: Base class for all market environments
- OrderBookEnv: Limit order book environment (HFT)
- OHLCVEnv: OHLCV bar environment (MFT/LFT)
- MarketDataGenerator: Synthetic and real data generation

Each agent wraps these base environments with their own
frequency-specific logic and reward functions.
"""

from .base_env import BaseMarketEnv
from .orderbook_env import OrderBookEnv
from .ohlcv_env import OHLCVEnv
from .data_generator import MarketDataGenerator

__all__ = [
    'BaseMarketEnv',
    'OrderBookEnv',
    'OHLCVEnv',
    'MarketDataGenerator'
]

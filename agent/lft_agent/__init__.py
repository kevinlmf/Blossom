"""
LFT Agent Module for Low-Frequency Portfolio Optimization

Structure:
├── env/       - Multi-asset environment wrapper for portfolio optimization
├── data/      - Daily data loader with macro features and factor exposures
└── strategy/  - LFT networks, SAC agent for portfolio optimization

Goal: Maintain long-term balance and reduce portfolio drawdown
Reward: r_LFT = R_portfolio - λ * CVaR(R_total)
"""

from .env.lft_env_wrapper import LFTEnvWrapper
from .data.daily_data_loader import DailyDataLoader
from .strategy.networks import LFTActor, LFTCritic, LFTActorCritic
from .strategy.sac_agent import LFTSACAgent
from .strategy.replay_buffer import ReplayBuffer as LFTReplayBuffer
from .selection.lft_stock_selector import LFTStockSelector

__all__ = [
    'LFTEnvWrapper', 'DailyDataLoader',
    'LFTActor', 'LFTCritic', 'LFTActorCritic', 'LFTSACAgent',
    'LFTReplayBuffer', 'LFTStockSelector'
]

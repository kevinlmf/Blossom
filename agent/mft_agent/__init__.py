"""
MFT Agent Module for Medium-Frequency Trading

Structure:
├── env/       - OHLCV environment wrapper for hourly/daily trading
├── data/      - Hourly/daily data loader with technical indicators
└── strategy/  - MFT networks, SAC agent, and replay buffer

Goal: Hedge HFT volatility while exploiting medium-term trends
Reward: r_MFT = -ρ * Corr(HFT, MFT) + η * E[R_MFT]
"""

from .env.mft_env_wrapper import MFTEnvWrapper
from .data.hourly_data_loader import HourlyDataLoader
from .strategy.networks import MFTActor, MFTCritic, MFTActorCritic
from .strategy.sac_agent import MFTSACAgent

__all__ = [
    'MFTEnvWrapper', 'HourlyDataLoader',
    'MFTActor', 'MFTCritic', 'MFTActorCritic', 'MFTSACAgent'
]

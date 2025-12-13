"""
Allocator Agent Module (Meta Level)

Purpose:
- Meta-level capital allocation across HFT/MFT/LFT agents
- Maximize long-term wealth, Sharpe ratio, minimize CVaR

Input:
- Macro indicators (volatility, liquidity, regime)
- Performance signals from HFT/MFT/LFT agents
- Latent factors z_t from Shared Encoder

Output:
- Capital allocation ratios [π_HFT, π_MFT, π_LFT]

Objective:
- Maximize long-term wealth
- Optimize Sharpe ratio
- Minimize Conditional Value at Risk (CVaR)

Training:
- PPO-based reinforcement learning
- Learns optimal allocation policy
"""

from .allocator import CapitalAllocator
from .ppo_allocator import PPOAllocatorAgent
from .transfer_ppo_allocator import (
    TransferPPOAllocator,
    DomainAdaptiveAllocator,
    create_transfer_allocator
)

__all__ = [
    'CapitalAllocator',
    'PPOAllocatorAgent',
    'TransferPPOAllocator',
    'DomainAdaptiveAllocator',
    'create_transfer_allocator'
]

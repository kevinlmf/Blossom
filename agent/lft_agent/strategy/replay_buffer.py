"""
Replay Buffer for LFT Agent

Reuses the replay buffer implementation from HFT agent.
"""

# Import from HFT agent to avoid code duplication
import sys
from pathlib import Path

# Add HFT agent to path
hft_path = Path(__file__).parent.parent.parent / "hft_agent" / "strategy"
sys.path.insert(0, str(hft_path))

from replay_buffer import ReplayBuffer, Transition, PrioritizedReplayBuffer, SequenceReplayBuffer

__all__ = ['ReplayBuffer', 'Transition', 'PrioritizedReplayBuffer', 'SequenceReplayBuffer']

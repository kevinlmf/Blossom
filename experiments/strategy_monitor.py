"""
Strategy Performance Monitor

Tracks and logs performance metrics for each trading agent and the overall system.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from datetime import datetime


class StrategyMonitor:
    """
    Monitor for tracking strategy performance across different agents and market regimes.
    """

    def __init__(
        self,
        output_dir: str = "outputs/monitoring",
        window_size: int = 100,
        verbose: bool = True
    ):
        """
        Initialize the strategy monitor.

        Args:
            output_dir: Directory to save monitoring outputs
            window_size: Window size for rolling statistics
            verbose: Whether to print detailed logs
        """
        self.output_dir = output_dir
        self.window_size = window_size
        self.verbose = verbose

        os.makedirs(output_dir, exist_ok=True)

        # Metrics storage
        self.episode_metrics = defaultdict(list)
        self.step_metrics = defaultdict(lambda: deque(maxlen=window_size))

        # Agent-specific metrics
        self.agent_returns = defaultdict(list)
        self.agent_sharpe = defaultdict(list)
        self.agent_drawdown = defaultdict(list)

        # System-level metrics
        self.total_capital_history = []
        self.allocation_history = []
        self.risk_scores = []

        # Current episode tracking
        self.current_episode = 0
        self.episode_start_time = None

        if self.verbose:
            print(f"Strategy Monitor initialized. Output: {output_dir}")

    def start_episode(self, episode: int):
        """Start tracking a new episode."""
        self.current_episode = episode
        self.episode_start_time = datetime.now()

        # Reset step-level metrics for new episode
        for key in self.step_metrics.keys():
            self.step_metrics[key].clear()

    def log_step(
        self,
        step: int,
        hft_reward: float,
        mft_reward: float,
        lft_reward: float,
        allocator_reward: float,
        capital: Dict[str, float],
        allocation: np.ndarray,
        risk_score: float
    ):
        """
        Log metrics for a single step.

        Args:
            step: Current step number
            hft_reward: HFT agent reward
            mft_reward: MFT agent reward
            lft_reward: LFT agent reward
            allocator_reward: Allocator reward
            capital: Capital allocation per agent
            allocation: Current allocation weights
            risk_score: Current risk score
        """
        # Store step metrics
        self.step_metrics['hft_reward'].append(hft_reward)
        self.step_metrics['mft_reward'].append(mft_reward)
        self.step_metrics['lft_reward'].append(lft_reward)
        self.step_metrics['allocator_reward'].append(allocator_reward)

        # Store capital and allocation
        total_capital = sum(capital.values())
        self.step_metrics['total_capital'].append(total_capital)
        self.step_metrics['risk_score'].append(risk_score)

    def end_episode(
        self,
        episode: int,
        episode_metrics: Dict[str, float],
        capital: Dict[str, float],
        final_allocation: np.ndarray
    ):
        """
        End episode tracking and compute summary statistics.

        Args:
            episode: Episode number
            episode_metrics: Cumulative metrics for the episode
            capital: Final capital per agent
            final_allocation: Final allocation weights
        """
        # Compute episode statistics
        episode_duration = (datetime.now() - self.episode_start_time).total_seconds()

        # Store episode metrics
        self.episode_metrics['episode'].append(episode)
        self.episode_metrics['duration'].append(episode_duration)

        for agent in ['hft', 'mft', 'lft', 'allocator']:
            reward = episode_metrics.get(f'{agent}_reward', 0.0)
            self.episode_metrics[f'{agent}_reward'].append(reward)

        # Store capital and allocation
        total_capital = sum(capital.values())
        self.total_capital_history.append(total_capital)
        self.allocation_history.append(final_allocation.tolist())

        # Compute Sharpe ratios (if enough data)
        if len(self.agent_returns['hft']) >= 10:
            for agent in ['hft', 'mft', 'lft']:
                returns = np.array(self.agent_returns[agent][-self.window_size:])
                sharpe = self._compute_sharpe_ratio(returns)
                self.agent_sharpe[agent].append(sharpe)

        # Save periodic snapshots
        if (episode + 1) % 100 == 0:
            self.save_metrics()
            if self.verbose:
                print(f"\nMetrics saved for episode {episode + 1}")

    def log_market_regime(self, regime: str, regime_metrics: Dict[str, float]):
        """
        Log performance for a specific market regime.

        Args:
            regime: Market regime ('high_risk', 'high_return', 'stable')
            regime_metrics: Performance metrics for this regime
        """
        regime_key = f'regime_{regime}'

        for key, value in regime_metrics.items():
            self.episode_metrics[f'{regime_key}_{key}'].append(value)

    def _compute_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Compute Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < 1e-8:
            return 0.0

        sharpe = (mean_return - risk_free_rate) / std_return
        return float(sharpe)

    def _compute_max_drawdown(self, capital_series: np.ndarray) -> float:
        """Compute maximum drawdown from capital series."""
        if len(capital_series) < 2:
            return 0.0

        cummax = np.maximum.accumulate(capital_series)
        drawdown = (capital_series - cummax) / cummax
        max_drawdown = np.min(drawdown)

        return float(max_drawdown)

    def save_metrics(self, filename: Optional[str] = None):
        """Save metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        # Prepare data for JSON serialization
        data = {
            'episode_metrics': {k: v for k, v in self.episode_metrics.items()},
            'total_capital_history': self.total_capital_history,
            'allocation_history': self.allocation_history,
            'agent_sharpe': {k: v for k, v in self.agent_sharpe.items()},
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"Metrics saved to: {filepath}")

    def plot_performance(self, save_path: Optional[str] = None):
        """
        Plot performance metrics.

        Args:
            save_path: Optional path to save the plot
        """
        if len(self.episode_metrics['episode']) < 2:
            print("Not enough data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Agent Rewards
        episodes = self.episode_metrics['episode']
        axes[0, 0].plot(episodes, self.episode_metrics['hft_reward'], label='HFT', alpha=0.7)
        axes[0, 0].plot(episodes, self.episode_metrics['mft_reward'], label='MFT', alpha=0.7)
        axes[0, 0].plot(episodes, self.episode_metrics['lft_reward'], label='LFT', alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].set_title('Agent Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Total Capital
        axes[0, 1].plot(episodes, self.total_capital_history, color='green', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Capital ($)')
        axes[0, 1].set_title('Total Capital Over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Allocation Over Time
        if len(self.allocation_history) > 0:
            allocations = np.array(self.allocation_history)
            axes[1, 0].plot(episodes, allocations[:, 0], label='HFT', alpha=0.7)
            axes[1, 0].plot(episodes, allocations[:, 1], label='MFT', alpha=0.7)
            axes[1, 0].plot(episodes, allocations[:, 2], label='LFT', alpha=0.7)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Allocation Weight')
            axes[1, 0].set_title('Capital Allocation Over Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Sharpe Ratios
        if len(self.agent_sharpe['hft']) > 0:
            sharpe_episodes = episodes[-len(self.agent_sharpe['hft']):]
            axes[1, 1].plot(sharpe_episodes, self.agent_sharpe['hft'], label='HFT', alpha=0.7)
            axes[1, 1].plot(sharpe_episodes, self.agent_sharpe['mft'], label='MFT', alpha=0.7)
            axes[1, 1].plot(sharpe_episodes, self.agent_sharpe['lft'], label='LFT', alpha=0.7)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].set_title('Agent Sharpe Ratios')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"performance_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"Performance plot saved to: {save_path}")

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for the current monitoring session."""
        if len(self.total_capital_history) < 2:
            return {}

        stats = {
            'total_episodes': len(self.episode_metrics['episode']),
            'final_capital': self.total_capital_history[-1],
            'initial_capital': self.total_capital_history[0] if self.total_capital_history else 0.0,
            'total_return': (self.total_capital_history[-1] / self.total_capital_history[0] - 1) * 100
            if self.total_capital_history[0] > 0 else 0.0,
            'max_drawdown': self._compute_max_drawdown(np.array(self.total_capital_history)),
        }

        # Agent-specific stats
        for agent in ['hft', 'mft', 'lft']:
            if len(self.agent_sharpe[agent]) > 0:
                stats[f'{agent}_avg_sharpe'] = np.mean(self.agent_sharpe[agent])

        return stats


def create_monitor(
    output_dir: str = "outputs/monitoring",
    window_size: int = 100,
    verbose: bool = True
) -> StrategyMonitor:
    """
    Factory function to create a StrategyMonitor instance.

    Args:
        output_dir: Directory to save monitoring outputs
        window_size: Window size for rolling statistics
        verbose: Whether to print detailed logs

    Returns:
        StrategyMonitor instance
    """
    return StrategyMonitor(
        output_dir=output_dir,
        window_size=window_size,
        verbose=verbose
    )

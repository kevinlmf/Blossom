"""
Base Market Environment for Multi-Frequency Trading System
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
import chex
from abc import ABC, abstractmethod


class BaseMarketEnv(ABC):
    """
    Abstract base class for market environments.

    Provides common interface for all frequency levels:
    - High-Frequency (tick-level, orderbook)
    - Medium-Frequency (hourly, daily)
    - Low-Frequency (daily, weekly)
    """

    def __init__(
        self,
        num_assets: int,
        initial_capital: float = 1_000_000.0,
        transaction_cost: float = 0.0001,
        slippage: float = 0.0005,
        market_impact: float = 0.0002
    ):
        """
        Initialize base market environment.

        Args:
            num_assets: Number of tradable assets
            initial_capital: Initial capital for trading
            transaction_cost: Transaction cost (fraction of trade value)
            slippage: Slippage cost (fraction of trade value)
            market_impact: Market impact cost (fraction of trade value)
        """
        self.num_assets = num_assets
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.market_impact = market_impact

        # State variables
        self.current_step = 0
        self.done = False
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = jnp.zeros(num_assets)

    @abstractmethod
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            key: JAX random key

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        pass

    @abstractmethod
    def step(
        self,
        action: chex.Array
    ) -> Tuple[chex.Array, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Trading action

        Returns:
            observation: Next observation
            reward: Reward signal
            done: Episode termination flag
            info: Additional information dictionary
        """
        pass

    @abstractmethod
    def get_observation(self) -> chex.Array:
        """
        Get current observation.

        Returns:
            observation: Current state observation
        """
        pass

    def compute_transaction_costs(
        self,
        trade_value: chex.Array
    ) -> chex.Array:
        """
        Compute total transaction costs.

        Args:
            trade_value: Absolute value of trades

        Returns:
            total_cost: Total transaction cost
        """
        commission = trade_value * self.transaction_cost
        slippage_cost = trade_value * self.slippage
        impact_cost = trade_value * self.market_impact

        total_cost = commission + slippage_cost + impact_cost
        return total_cost

    def update_portfolio(
        self,
        prices: chex.Array,
        trades: chex.Array
    ) -> Tuple[float, float, chex.Array]:
        """
        Update portfolio state after trades.

        Args:
            prices: Current asset prices [num_assets]
            trades: Number of shares traded [num_assets]

        Returns:
            new_cash: Updated cash position
            transaction_costs: Total transaction costs
            new_positions: Updated positions
        """
        # Compute trade values
        trade_value = jnp.abs(trades * prices)
        total_trade_value = jnp.sum(trade_value)

        # Compute transaction costs
        transaction_costs = self.compute_transaction_costs(total_trade_value)

        # Update positions and cash
        new_positions = self.positions + trades
        cash_flow = -jnp.sum(trades * prices)  # Negative for buys, positive for sells
        new_cash = self.cash + cash_flow - transaction_costs

        return new_cash, transaction_costs, new_positions

    def get_portfolio_value(
        self,
        prices: chex.Array
    ) -> float:
        """
        Compute current portfolio value.

        Args:
            prices: Current asset prices

        Returns:
            portfolio_value: Total portfolio value (cash + positions)
        """
        position_value = jnp.sum(self.positions * prices)
        portfolio_value = self.cash + position_value
        return portfolio_value

    def get_portfolio_weights(
        self,
        prices: chex.Array
    ) -> chex.Array:
        """
        Compute current portfolio weights.

        Args:
            prices: Current asset prices

        Returns:
            weights: Portfolio weights [num_assets]
        """
        portfolio_value = self.get_portfolio_value(prices)
        position_values = self.positions * prices
        weights = position_values / (portfolio_value + 1e-8)
        return weights

    def compute_returns(
        self,
        prev_value: float,
        current_value: float
    ) -> float:
        """
        Compute portfolio returns.

        Args:
            prev_value: Previous portfolio value
            current_value: Current portfolio value

        Returns:
            returns: Portfolio returns
        """
        returns = (current_value - prev_value) / (prev_value + 1e-8)
        return returns

    def check_constraints(
        self,
        action: chex.Array,
        prices: chex.Array
    ) -> Tuple[bool, str]:
        """
        Check if action satisfies constraints.

        Args:
            action: Proposed action
            prices: Current prices

        Returns:
            valid: Whether action is valid
            message: Constraint violation message
        """
        # Compute resulting positions
        trades = self._action_to_trades(action, prices)
        new_positions = self.positions + trades

        # Check cash constraint
        cash_needed = jnp.sum(jnp.maximum(0, trades) * prices)
        if cash_needed > self.cash:
            return False, "Insufficient cash"

        # Check short-selling constraint (if applicable)
        if jnp.any(new_positions < 0):
            return False, "Short-selling not allowed"

        return True, ""

    @abstractmethod
    def _action_to_trades(
        self,
        action: chex.Array,
        prices: chex.Array
    ) -> chex.Array:
        """
        Convert action to number of shares to trade.

        Args:
            action: Agent action
            prices: Current prices

        Returns:
            trades: Number of shares to trade for each asset
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get environment information.

        Returns:
            info: Dictionary of environment information
        """
        return {
            'step': self.current_step,
            'portfolio_value': float(self.portfolio_value),
            'cash': float(self.cash),
            'positions': self.positions,
            'done': self.done
        }


class MultiFrequencyEnv(BaseMarketEnv):
    """
    Multi-frequency environment that integrates HFT, MFT, and LFT.

    This environment provides:
    - Tick-level data for HFT
    - Hourly/daily data for MFT
    - Daily/weekly data for LFT
    """

    def __init__(
        self,
        num_assets: int,
        hft_data: Optional[chex.Array] = None,
        mft_data: Optional[chex.Array] = None,
        lft_data: Optional[chex.Array] = None,
        **kwargs
    ):
        """
        Initialize multi-frequency environment.

        Args:
            num_assets: Number of assets
            hft_data: High-frequency tick data
            mft_data: Medium-frequency (hourly/daily) data
            lft_data: Low-frequency (daily/weekly) data
            **kwargs: Additional arguments for base class
        """
        super().__init__(num_assets, **kwargs)

        self.hft_data = hft_data
        self.mft_data = mft_data
        self.lft_data = lft_data

        # Frequency counters
        self.hft_step = 0
        self.mft_step = 0
        self.lft_step = 0

        # Frequency ratios (e.g., 1 MFT step = 60 HFT steps)
        self.hft_to_mft_ratio = 60
        self.mft_to_lft_ratio = 24

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, Any]]:
        """Reset environment."""
        self.current_step = 0
        self.hft_step = 0
        self.mft_step = 0
        self.lft_step = 0
        self.done = False

        self.cash = self.initial_capital
        self.positions = jnp.zeros(self.num_assets)
        self.portfolio_value = self.initial_capital

        observation = self.get_observation()
        info = self.get_info()

        return observation, info

    def step(
        self,
        action: chex.Array
    ) -> Tuple[chex.Array, float, bool, Dict[str, Any]]:
        """Execute one step."""
        # This will be implemented based on specific frequency
        raise NotImplementedError("Use frequency-specific environments")

    def get_observation(self) -> chex.Array:
        """Get multi-frequency observation."""
        # Combine observations from all frequencies
        observations = []

        if self.hft_data is not None:
            hft_obs = self.hft_data[self.hft_step]
            observations.append(hft_obs)

        if self.mft_data is not None:
            mft_obs = self.mft_data[self.mft_step]
            observations.append(mft_obs)

        if self.lft_data is not None:
            lft_obs = self.lft_data[self.lft_step]
            observations.append(lft_obs)

        return jnp.concatenate(observations)

    def _action_to_trades(
        self,
        action: chex.Array,
        prices: chex.Array
    ) -> chex.Array:
        """Convert action to trades."""
        # Simple implementation: action directly represents target positions
        target_positions = action * self.portfolio_value / prices
        trades = target_positions - self.positions
        return trades

    def sync_frequencies(self) -> Dict[str, bool]:
        """
        Check which frequency levels need to be updated.

        Returns:
            Dictionary indicating which frequencies to update
        """
        update_hft = True  # Always update HFT
        update_mft = (self.hft_step % self.hft_to_mft_ratio == 0)
        update_lft = (self.mft_step % self.mft_to_lft_ratio == 0) and update_mft

        return {
            'hft': update_hft,
            'mft': update_mft,
            'lft': update_lft
        }

"""
OHLCV Environment for Medium and Low-Frequency Trading
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
import chex
from .base_env import BaseMarketEnv


class OHLCVEnv(BaseMarketEnv):
    """
    OHLCV (Open-High-Low-Close-Volume) Environment for MFT and LFT.

    Features:
    - Multiple timeframes (hourly, daily, weekly)
    - Technical indicators
    - Multi-asset support
    - Macro features integration
    """

    def __init__(
        self,
        num_assets: int,
        lookback_window: int = 100,
        max_steps: int = 1000,
        frequency: str = "daily",  # hourly, daily, weekly
        price_data: Optional[chex.Array] = None,
        volume_data: Optional[chex.Array] = None,
        feature_data: Optional[chex.Array] = None,
        **kwargs
    ):
        """
        Initialize OHLCV environment.

        Args:
            num_assets: Number of tradable assets
            lookback_window: Number of historical bars to include in observation
            max_steps: Maximum steps per episode
            frequency: Trading frequency (hourly, daily, weekly)
            price_data: Historical price data [T, num_assets, 4] (OHLC)
            volume_data: Historical volume data [T, num_assets]
            feature_data: Additional features [T, num_features]
            **kwargs: Additional arguments for base class
        """
        super().__init__(num_assets, **kwargs)

        self.lookback_window = lookback_window
        self.max_steps = max_steps
        self.frequency = frequency

        # Market data
        self.price_data = price_data
        self.volume_data = volume_data
        self.feature_data = feature_data

        # Current state
        self.current_index = lookback_window
        self.prices = None
        self.returns_history = []

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, Any]]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.current_index = self.lookback_window
        self.done = False

        # Reset portfolio
        self.cash = self.initial_capital
        self.positions = jnp.zeros(self.num_assets)
        self.portfolio_value = self.initial_capital

        # Get initial prices
        self.prices = self._get_current_prices()

        # Reset history
        self.returns_history = []

        observation = self.get_observation()
        info = self.get_info()

        return observation, info

    def step(
        self,
        action: chex.Array
    ) -> Tuple[chex.Array, float, bool, Dict[str, Any]]:
        """
        Execute one step in the OHLCV environment.

        Args:
            action: Portfolio weights [num_assets]

        Returns:
            observation: Next observation
            reward: Reward signal
            done: Episode termination flag
            info: Additional information
        """
        # Store previous portfolio value
        prev_value = self.portfolio_value

        # Convert action (weights) to trades
        current_prices = self._get_current_prices()
        trades = self._action_to_trades(action, current_prices)

        # Update portfolio
        self.cash, transaction_costs, self.positions = self.update_portfolio(
            current_prices, trades
        )

        # Move to next time step
        self.current_index += 1
        self.current_step += 1

        # Get new prices
        new_prices = self._get_current_prices()

        # Compute portfolio value and returns
        self.portfolio_value = self.get_portfolio_value(new_prices)
        portfolio_return = self.compute_returns(prev_value, self.portfolio_value)
        self.returns_history.append(portfolio_return)

        # Compute reward
        reward = self._compute_reward(
            portfolio_return,
            transaction_costs,
            action
        )

        # Check if episode is done
        price_data_exhausted = (self.price_data is not None and
                                self.current_index >= len(self.price_data) - 1)
        self.done = (
            price_data_exhausted or
            self.current_step >= self.max_steps or
            self.portfolio_value <= self.initial_capital * 0.5  # 50% drawdown limit
        )

        # Get next observation
        observation = self.get_observation()

        info = self.get_info()
        info.update({
            'portfolio_return': float(portfolio_return),
            'transaction_costs': float(transaction_costs),
            'sharpe_ratio': float(self._compute_sharpe_ratio()),
            'max_drawdown': float(self._compute_max_drawdown())
        })

        return observation, float(reward), self.done, info

    def _get_current_prices(self) -> chex.Array:
        """Get current closing prices."""
        if self.price_data is not None:
            # Return close prices (index 3 in OHLC)
            return self.price_data[self.current_index, :, 3]
        else:
            # Generate random prices for testing
            return jnp.ones(self.num_assets) * 100.0

    def _action_to_trades(
        self,
        action: chex.Array,
        prices: chex.Array
    ) -> chex.Array:
        """
        Convert portfolio weights to number of shares to trade.

        Args:
            action: Target portfolio weights [num_assets]
            prices: Current prices [num_assets]

        Returns:
            trades: Number of shares to trade [num_assets]
        """
        # Normalize weights to sum to 1
        weights = action / (jnp.sum(jnp.abs(action)) + 1e-8)
        weights = jnp.clip(weights, 0, 1)  # Long-only constraint
        weights = weights / (jnp.sum(weights) + 1e-8)

        # Compute target positions
        target_value = weights * self.portfolio_value
        target_positions = target_value / (prices + 1e-8)

        # Compute trades
        trades = target_positions - self.positions

        return trades

    def _compute_reward(
        self,
        portfolio_return: float,
        transaction_costs: float,
        action: chex.Array
    ) -> float:
        """
        Compute reward for MFT/LFT agent.

        Reward components:
        1. Portfolio returns
        2. Risk-adjusted returns (Sharpe)
        3. Transaction cost penalty
        4. Concentration penalty
        """
        # Base reward: portfolio returns
        reward = portfolio_return

        # Risk-adjusted returns (if enough history)
        if len(self.returns_history) >= 30:
            sharpe_ratio = self._compute_sharpe_ratio()
            reward += 0.1 * sharpe_ratio

        # Transaction cost penalty
        cost_penalty = -transaction_costs / self.portfolio_value
        reward += cost_penalty

        # Concentration penalty (encourage diversification)
        concentration = jnp.sum(action ** 2)
        concentration_penalty = -0.01 * concentration
        reward += concentration_penalty

        return reward

    def _compute_sharpe_ratio(self, window: int = 30) -> float:
        """Compute Sharpe ratio over recent history."""
        if len(self.returns_history) < window:
            return 0.0

        recent_returns = jnp.array(self.returns_history[-window:])
        mean_return = jnp.mean(recent_returns)
        std_return = jnp.std(recent_returns)

        sharpe = mean_return / (std_return + 1e-8)
        return sharpe * jnp.sqrt(252)  # Annualized

    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown."""
        if len(self.returns_history) == 0:
            return 0.0

        returns_array = jnp.array([1.0] + self.returns_history)
        cumulative_returns = jnp.cumprod(1 + returns_array)

        running_max = jnp.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        max_drawdown = jnp.min(drawdown)
        return max_drawdown

    def get_observation(self) -> chex.Array:
        """
        Get current observation including:
        - Historical OHLCV data
        - Technical indicators
        - Portfolio state
        - Additional features
        """
        observations = []

        # 1. Historical price data (returns)
        if self.price_data is not None:
            start_idx = self.current_index - self.lookback_window
            end_idx = self.current_index

            # Get OHLC data
            historical_prices = self.price_data[start_idx:end_idx]  # [window, assets, 4]

            # Compute returns
            close_prices = historical_prices[:, :, 3]  # [window, assets]
            returns = jnp.diff(jnp.log(close_prices + 1e-8), axis=0)  # [window-1, assets]
            returns = jnp.pad(returns, ((1, 0), (0, 0)), constant_values=0)

            # Flatten and normalize
            returns_flat = returns.flatten() / 0.02  # Normalize by typical daily vol

            observations.append(returns_flat)

        # 2. Volume data
        if self.volume_data is not None:
            start_idx = self.current_index - self.lookback_window
            end_idx = self.current_index

            historical_volumes = self.volume_data[start_idx:end_idx]
            # Normalize by rolling mean
            volume_mean = jnp.mean(historical_volumes, axis=0, keepdims=True)
            volume_normalized = historical_volumes / (volume_mean + 1e-8)

            observations.append(volume_normalized.flatten())

        # 3. Technical indicators
        technical_features = self._compute_technical_indicators()
        observations.append(technical_features)

        # 4. Portfolio state
        current_prices = self._get_current_prices()
        portfolio_weights = self.get_portfolio_weights(current_prices)
        portfolio_state = jnp.concatenate([
            portfolio_weights,
            jnp.array([self.cash / self.portfolio_value]),
            jnp.array([len(self.returns_history) / self.max_steps])  # Progress
        ])
        observations.append(portfolio_state)

        # 5. Additional features
        if self.feature_data is not None:
            additional_features = self.feature_data[self.current_index]
            observations.append(additional_features)

        # Concatenate all observations
        observation = jnp.concatenate(observations)

        return observation

    def _compute_technical_indicators(self) -> chex.Array:
        """Compute technical indicators (RSI, MACD, Bollinger Bands, etc.)."""
        if self.price_data is None:
            return jnp.zeros(self.num_assets * 3)

        # Get recent price data
        lookback = min(50, self.current_index)
        prices = self.price_data[self.current_index - lookback:self.current_index, :, 3]

        indicators = []

        for asset_idx in range(self.num_assets):
            asset_prices = prices[:, asset_idx]

            # 1. RSI (Relative Strength Index)
            rsi = self._compute_rsi(asset_prices, period=14)
            indicators.append(rsi / 100.0)  # Normalize to [0, 1]

            # 2. Moving average ratio
            ma_short = jnp.mean(asset_prices[-10:])
            ma_long = jnp.mean(asset_prices[-30:]) if len(asset_prices) >= 30 else ma_short
            ma_ratio = ma_short / (ma_long + 1e-8) - 1.0
            indicators.append(ma_ratio)

            # 3. Volatility (normalized)
            returns = jnp.diff(jnp.log(asset_prices + 1e-8))
            volatility = jnp.std(returns)
            indicators.append(volatility / 0.02)

        return jnp.array(indicators)

    def _compute_rsi(self, prices: chex.Array, period: int = 14) -> float:
        """Compute Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0

        # Compute price changes
        deltas = jnp.diff(prices)

        # Separate gains and losses
        gains = jnp.where(deltas > 0, deltas, 0)
        losses = jnp.where(deltas < 0, -deltas, 0)

        # Compute average gains and losses
        avg_gain = jnp.mean(gains[-period:])
        avg_loss = jnp.mean(losses[-period:])

        # Compute RS and RSI
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def set_data(
        self,
        price_data: chex.Array,
        volume_data: Optional[chex.Array] = None,
        feature_data: Optional[chex.Array] = None
    ) -> None:
        """Set market data for the environment."""
        self.price_data = price_data
        self.volume_data = volume_data
        self.feature_data = feature_data


class MultiAssetOHLCVEnv(OHLCVEnv):
    """Extended OHLCV environment with multi-asset correlation features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_correlation_features(self) -> chex.Array:
        """Compute cross-asset correlation features."""
        if self.price_data is None or self.current_index < 30:
            return jnp.zeros(self.num_assets)

        # Get recent returns
        lookback = 30
        prices = self.price_data[self.current_index - lookback:self.current_index, :, 3]
        returns = jnp.diff(jnp.log(prices + 1e-8), axis=0)

        # Compute correlation matrix
        correlation_matrix = jnp.corrcoef(returns.T)

        # Extract mean correlation for each asset
        mean_correlations = jnp.mean(correlation_matrix, axis=1)

        return mean_correlations

    def get_observation(self) -> chex.Array:
        """Extended observation with correlation features."""
        base_observation = super().get_observation()
        correlation_features = self._compute_correlation_features()

        return jnp.concatenate([base_observation, correlation_features])

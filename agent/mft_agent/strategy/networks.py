"""
Neural Network architectures for MFT Agent (Medium-Frequency Trading)

MFT operates on hourly/daily OHLCV data and focuses on:
- Hedging HFT volatility
- Exploiting medium-term trends
- Momentum and mean-reversion strategies
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Sequence, Optional
import chex


class MFTActor(nn.Module):
    """
    Actor network for MFT agent.

    Outputs:
    - Position size (continuous in [-1, 1])
    - Position type (long/short/neutral)
    """

    hidden_dims: Sequence[int] = (256, 256)
    action_dim: int = 2  # Position size + position type
    activation: str = "relu"

    @nn.compact
    def __call__(self, state: chex.Array, latent_factors: chex.Array) -> chex.Array:
        """
        Forward pass through actor network.

        Args:
            state: Market state (OHLCV features, indicators)
            latent_factors: Shared latent factors from encoder

        Returns:
            action: Trading action [position_size, position_type]
        """
        # Concatenate state and latent factors
        x = jnp.concatenate([state, latent_factors], axis=-1)

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            if self.activation == "relu":
                x = nn.relu(x)
            elif self.activation == "tanh":
                x = nn.tanh(x)
            elif self.activation == "gelu":
                x = nn.gelu(x)

        # Position size: tanh output scaled to [-1, 1]
        position_size = nn.Dense(1)(x)
        position_size = nn.tanh(position_size)

        # Position type: sigmoid for continuous position
        position_type = nn.Dense(1)(x)
        position_type = nn.sigmoid(position_type)

        # Concatenate outputs
        action = jnp.concatenate([position_size, position_type], axis=-1)

        return action


class MFTCritic(nn.Module):
    """
    Critic network for MFT agent (Q-function).

    Implements double Q-learning with twin Q-networks.
    """

    hidden_dims: Sequence[int] = (256, 256)
    activation: str = "relu"

    @nn.compact
    def __call__(
        self,
        state: chex.Array,
        action: chex.Array,
        latent_factors: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Forward pass through twin Q-networks.

        Args:
            state: Market state
            action: Trading action
            latent_factors: Shared latent factors

        Returns:
            q1, q2: Q-values from twin networks
        """
        # Concatenate inputs
        x = jnp.concatenate([state, action, latent_factors], axis=-1)

        # First Q-network
        q1 = x
        for hidden_dim in self.hidden_dims:
            q1 = nn.Dense(hidden_dim)(q1)
            if self.activation == "relu":
                q1 = nn.relu(q1)
            elif self.activation == "tanh":
                q1 = nn.tanh(q1)
        q1 = nn.Dense(1)(q1)

        # Second Q-network
        q2 = x
        for hidden_dim in self.hidden_dims:
            q2 = nn.Dense(hidden_dim)(q2)
            if self.activation == "relu":
                q2 = nn.relu(q2)
            elif self.activation == "tanh":
                q2 = nn.tanh(q2)
        q2 = nn.Dense(1)(q2)

        return jnp.squeeze(q1, axis=-1), jnp.squeeze(q2, axis=-1)


class MFTActorCritic(nn.Module):
    """Combined Actor-Critic network for MFT agent."""

    actor_hidden_dims: Sequence[int] = (256, 256)
    critic_hidden_dims: Sequence[int] = (256, 256)
    action_dim: int = 2
    activation: str = "relu"

    def setup(self):
        """Initialize actor and critic networks."""
        self.actor = MFTActor(
            hidden_dims=self.actor_hidden_dims,
            action_dim=self.action_dim,
            activation=self.activation
        )

        self.critic = MFTCritic(
            hidden_dims=self.critic_hidden_dims,
            activation=self.activation
        )

    def __call__(
        self,
        state: chex.Array,
        action: Optional[chex.Array],
        latent_factors: chex.Array,
        mode: str = "actor"
    ) -> chex.Array:
        """Forward pass through network."""
        if mode == "actor":
            return self.actor(state, latent_factors)
        elif mode == "critic":
            if action is None:
                raise ValueError("Action must be provided for critic mode")
            return self.critic(state, action, latent_factors)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class OHLCVFeatureExtractor(nn.Module):
    """
    Feature extractor for OHLCV data.

    Computes technical indicators:
    - Moving averages (SMA, EMA)
    - Volatility (standard deviation, ATR)
    - Momentum (RSI, MACD)
    - Volume indicators
    """

    feature_dim: int = 64

    @nn.compact
    def __call__(
        self,
        open_prices: chex.Array,
        high_prices: chex.Array,
        low_prices: chex.Array,
        close_prices: chex.Array,
        volumes: chex.Array
    ) -> chex.Array:
        """
        Extract features from OHLCV data.

        Args:
            open_prices: Opening prices [lookback_window]
            high_prices: High prices [lookback_window]
            low_prices: Low prices [lookback_window]
            close_prices: Closing prices [lookback_window]
            volumes: Trading volumes [lookback_window]

        Returns:
            features: Extracted features [feature_dim]
        """
        # Basic price features
        current_price = close_prices[-1]
        price_change = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
        high_low_spread = (high_prices[-1] - low_prices[-1]) / current_price

        # Simple moving averages
        sma_5 = jnp.mean(close_prices[-5:])
        sma_20 = jnp.mean(close_prices[-20:]) if len(close_prices) >= 20 else sma_5

        # Volatility
        returns = jnp.diff(close_prices) / close_prices[:-1]
        volatility = jnp.std(returns)

        # Momentum
        momentum_5 = (close_prices[-1] - close_prices[-5]) / close_prices[-5]

        # Volume indicators
        avg_volume = jnp.mean(volumes)
        volume_ratio = volumes[-1] / (avg_volume + 1e-8)

        # Concatenate all features
        raw_features = jnp.concatenate([
            jnp.array([
                current_price / 100.0,  # Normalize
                price_change,
                high_low_spread,
                (current_price - sma_5) / sma_5,
                (current_price - sma_20) / sma_20,
                volatility,
                momentum_5,
                volume_ratio
            ]),
            # Include recent price history (normalized)
            (close_prices[-10:] / current_price)[:10]  # Last 10 periods
        ])

        # Project to feature dimension
        features = nn.Dense(self.feature_dim)(raw_features)
        features = nn.relu(features)

        return features


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for sequence modeling.

    Learns to attend to important time steps in price history.
    """

    num_heads: int = 4
    hidden_dim: int = 128

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        train: bool = True
    ) -> chex.Array:
        """
        Apply temporal attention.

        Args:
            x: Input sequence [batch, seq_len, features]
            train: Training mode

        Returns:
            output: Attended features [batch, hidden_dim]
        """
        # Multi-head self-attention
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim
        )(x, x)

        # Layer normalization
        attended = nn.LayerNorm()(attended)

        # Dropout
        attended = nn.Dropout(rate=0.1, deterministic=not train)(attended)

        # Global average pooling
        output = jnp.mean(attended, axis=1)

        return output


def create_mft_networks(
    state_dim: int,
    latent_dim: int,
    action_dim: int = 2,
    actor_hidden: Sequence[int] = (256, 256),
    critic_hidden: Sequence[int] = (256, 256),
    activation: str = "relu"
) -> MFTActorCritic:
    """
    Factory function to create MFT actor-critic networks.

    Args:
        state_dim: Dimension of state space
        latent_dim: Dimension of latent factors
        action_dim: Dimension of action space
        actor_hidden: Hidden layer sizes for actor
        critic_hidden: Hidden layer sizes for critic
        activation: Activation function

    Returns:
        actor_critic: MFTActorCritic network
    """
    return MFTActorCritic(
        actor_hidden_dims=actor_hidden,
        critic_hidden_dims=critic_hidden,
        action_dim=action_dim,
        activation=activation
    )

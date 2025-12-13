"""
Neural Network architectures for HFT Agent
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Sequence, Optional
import chex


class HFTActor(nn.Module):
    """
    Actor network for HFT agent.

    Outputs:
    - Order type (market/limit buy/sell)
    - Order quantity
    - Price offset (for limit orders)
    """

    hidden_dims: Sequence[int] = (256, 256)
    action_dim: int = 6  # [order_type (4), quantity (1), price_offset (1)]
    activation: str = "relu"

    @nn.compact
    def __call__(self, state: chex.Array, latent_factors: chex.Array) -> chex.Array:
        """
        Forward pass through actor network.

        Args:
            state: Market state (order book features)
            latent_factors: Shared latent factors from encoder

        Returns:
            action: Trading action
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

        # Output layers
        # Order type (softmax over 4 types)
        order_type_logits = nn.Dense(4)(x)
        order_type = nn.softmax(order_type_logits, axis=-1)

        # Order quantity (sigmoid, scaled to [0, 1])
        quantity_logits = nn.Dense(1)(x)
        quantity = nn.sigmoid(quantity_logits)

        # Price offset (tanh, scaled to [-1, 1])
        price_offset_logits = nn.Dense(1)(x)
        price_offset = nn.tanh(price_offset_logits)

        # Concatenate all action components
        action = jnp.concatenate([order_type, quantity, price_offset], axis=-1)

        return action


class HFTCritic(nn.Module):
    """
    Critic network for HFT agent (Q-function).

    Implements double Q-learning with two Q-networks.
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
        Forward pass through critic networks (twin Q-networks).

        Args:
            state: Market state
            action: Trading action
            latent_factors: Shared latent factors

        Returns:
            q1: Q-value from first network
            q2: Q-value from second network
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


class HFTActorCritic(nn.Module):
    """Combined Actor-Critic network for HFT agent."""

    actor_hidden_dims: Sequence[int] = (256, 256)
    critic_hidden_dims: Sequence[int] = (256, 256)
    action_dim: int = 6
    activation: str = "relu"

    def setup(self):
        """Initialize actor and critic networks."""
        self.actor = HFTActor(
            hidden_dims=self.actor_hidden_dims,
            action_dim=self.action_dim,
            activation=self.activation
        )

        self.critic = HFTCritic(
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
        """
        Forward pass through network.

        Args:
            state: Market state
            action: Trading action (required for critic mode)
            latent_factors: Shared latent factors
            mode: "actor" or "critic"

        Returns:
            Output based on mode
        """
        if mode == "actor":
            return self.actor(state, latent_factors)
        elif mode == "critic":
            if action is None:
                raise ValueError("Action must be provided for critic mode")
            return self.critic(state, action, latent_factors)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_action(
        self,
        state: chex.Array,
        latent_factors: chex.Array
    ) -> chex.Array:
        """Get action from actor."""
        return self.actor(state, latent_factors)

    def get_q_values(
        self,
        state: chex.Array,
        action: chex.Array,
        latent_factors: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Get Q-values from critic."""
        return self.critic(state, action, latent_factors)


class OrderBookFeatureExtractor(nn.Module):
    """
    Feature extractor for order book data.

    Processes raw order book into meaningful features:
    - Bid-ask imbalance
    - Price impact
    - Liquidity measures
    - Order flow toxicity
    """

    feature_dim: int = 64

    @nn.compact
    def __call__(
        self,
        bid_prices: chex.Array,
        bid_volumes: chex.Array,
        ask_prices: chex.Array,
        ask_volumes: chex.Array
    ) -> chex.Array:
        """
        Extract features from order book.

        Args:
            bid_prices: [depth_levels]
            bid_volumes: [depth_levels]
            ask_prices: [depth_levels]
            ask_volumes: [depth_levels]

        Returns:
            features: Extracted features [feature_dim]
        """
        # 1. Basic features
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        spread = ask_prices[0] - bid_prices[0]
        spread_pct = spread / mid_price

        # 2. Volume imbalance
        total_bid_volume = jnp.sum(bid_volumes)
        total_ask_volume = jnp.sum(ask_volumes)
        volume_imbalance = (total_bid_volume - total_ask_volume) / (
            total_bid_volume + total_ask_volume + 1e-8
        )

        # 3. Depth-weighted price
        bid_weighted_price = jnp.sum(bid_prices * bid_volumes) / (total_bid_volume + 1e-8)
        ask_weighted_price = jnp.sum(ask_prices * ask_volumes) / (total_ask_volume + 1e-8)

        # 4. Price levels
        price_spread_levels = jnp.concatenate([
            (mid_price - bid_prices) / mid_price,
            (ask_prices - mid_price) / mid_price
        ])

        # 5. Volume distribution
        bid_volume_dist = bid_volumes / (total_bid_volume + 1e-8)
        ask_volume_dist = ask_volumes / (total_ask_volume + 1e-8)

        # Concatenate all features
        raw_features = jnp.concatenate([
            jnp.array([mid_price / 100.0, spread_pct, volume_imbalance]),
            jnp.array([bid_weighted_price / 100.0, ask_weighted_price / 100.0]),
            price_spread_levels,
            bid_volume_dist,
            ask_volume_dist
        ])

        # Project to feature dimension
        features = nn.Dense(self.feature_dim)(raw_features)
        features = nn.relu(features)

        return features


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for processing order book time series.

    Uses dilated causal convolutions to capture temporal dependencies.
    """

    num_channels: Sequence[int] = (64, 128, 256)
    kernel_size: int = 3

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        train: bool = True
    ) -> chex.Array:
        """
        Forward pass through TCN.

        Args:
            x: Input time series [batch, seq_len, features]
            train: Training mode

        Returns:
            output: Processed features [batch, num_channels[-1]]
        """
        # Dilated causal convolutions
        for i, num_channel in enumerate(self.num_channels):
            dilation = 2 ** i

            # Causal padding
            padding = [(self.kernel_size - 1) * dilation, 0]

            x = nn.Conv(
                features=num_channel,
                kernel_size=(self.kernel_size,),
                kernel_dilation=(dilation,),
                padding=padding
            )(x)

            x = nn.relu(x)
            x = nn.Dropout(rate=0.1, deterministic=not train)(x)

        # Global average pooling
        output = jnp.mean(x, axis=1)

        return output


def create_hft_networks(
    state_dim: int,
    latent_dim: int,
    action_dim: int = 6,
    actor_hidden: Sequence[int] = (256, 256),
    critic_hidden: Sequence[int] = (256, 256),
    activation: str = "relu"
) -> HFTActorCritic:
    """
    Factory function to create HFT actor-critic networks.

    Args:
        state_dim: Dimension of state space
        latent_dim: Dimension of latent factors
        action_dim: Dimension of action space
        actor_hidden: Hidden layer sizes for actor
        critic_hidden: Hidden layer sizes for critic
        activation: Activation function

    Returns:
        actor_critic: HFTActorCritic network
    """
    return HFTActorCritic(
        actor_hidden_dims=actor_hidden,
        critic_hidden_dims=critic_hidden,
        action_dim=action_dim,
        activation=activation
    )

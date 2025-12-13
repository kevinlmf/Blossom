"""
Neural Network architectures for LFT Agent (Low-Frequency Trading)

LFT operates on daily/weekly data and focuses on:
- Long-term portfolio optimization
- Risk-adjusted returns (Sharpe ratio maximization)
- Minimizing drawdown and tail risk
- Asset allocation across multiple securities
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Sequence, Optional
import chex


class LFTActor(nn.Module):
    """
    Actor network for LFT agent.

    Outputs:
    - Portfolio weights (softmax over num_assets)
    - Rebalancing indicator
    """

    hidden_dims: Sequence[int] = (256, 256)
    num_assets: int = 10
    activation: str = "relu"

    @nn.compact
    def __call__(
        self,
        state: chex.Array,
        latent_factors: chex.Array
    ) -> chex.Array:
        """
        Forward pass through actor network.

        Args:
            state: Market state (returns, covariance, macro indicators)
            latent_factors: Shared latent factors from encoder

        Returns:
            action: Portfolio weights [num_assets]
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

        # Portfolio weights: softmax to ensure sum to 1
        logits = nn.Dense(self.num_assets)(x)
        weights = nn.softmax(logits, axis=-1)

        return weights


class LFTCritic(nn.Module):
    """
    Critic network for LFT agent (Value function).

    Evaluates expected long-term return for given portfolio allocation.
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
            action: Portfolio weights
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


class LFTActorCritic(nn.Module):
    """Combined Actor-Critic network for LFT agent."""

    actor_hidden_dims: Sequence[int] = (256, 256)
    critic_hidden_dims: Sequence[int] = (256, 256)
    num_assets: int = 10
    activation: str = "relu"

    def setup(self):
        """Initialize actor and critic networks."""
        self.actor = LFTActor(
            hidden_dims=self.actor_hidden_dims,
            num_assets=self.num_assets,
            activation=self.activation
        )

        self.critic = LFTCritic(
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


class PortfolioFeatureExtractor(nn.Module):
    """
    Feature extractor for portfolio optimization.

    Computes:
    - Expected returns (mean estimation)
    - Covariance matrix features
    - Risk metrics (volatility, VaR, CVaR)
    - Correlation structure
    """

    feature_dim: int = 128
    num_assets: int = 10

    @nn.compact
    def __call__(
        self,
        returns: chex.Array,  # [lookback, num_assets]
        prices: chex.Array    # [lookback, num_assets]
    ) -> chex.Array:
        """
        Extract portfolio features.

        Args:
            returns: Historical returns matrix
            prices: Historical prices matrix

        Returns:
            features: Extracted features [feature_dim]
        """
        # Expected returns (mean of recent returns)
        expected_returns = jnp.mean(returns, axis=0)

        # Volatilities
        volatilities = jnp.std(returns, axis=0)

        # Sharpe ratios (assuming zero risk-free rate)
        sharpe_ratios = expected_returns / (volatilities + 1e-8)

        # Correlation matrix
        corr_matrix = jnp.corrcoef(returns, rowvar=False)

        # Average correlation (portfolio diversification measure)
        n = self.num_assets
        avg_correlation = (jnp.sum(corr_matrix) - n) / (n * (n - 1))

        # Portfolio metrics (equal weight baseline)
        equal_weights = jnp.ones(self.num_assets) / self.num_assets
        portfolio_return = jnp.dot(equal_weights, expected_returns)
        portfolio_vol = jnp.sqrt(
            jnp.dot(equal_weights, jnp.dot(jnp.cov(returns, rowvar=False), equal_weights))
        )

        # Price momentum (trailing returns)
        momentum_1m = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else jnp.zeros(self.num_assets)
        momentum_3m = (prices[-1] - prices[-60]) / prices[-60] if len(prices) >= 60 else jnp.zeros(self.num_assets)

        # Concatenate all features
        raw_features = jnp.concatenate([
            expected_returns,
            volatilities,
            sharpe_ratios,
            momentum_1m,
            momentum_3m,
            jnp.array([avg_correlation, portfolio_return, portfolio_vol])
        ])

        # Project to feature dimension
        features = nn.Dense(self.feature_dim)(raw_features)
        features = nn.relu(features)

        return features


class CovarianceEstimator(nn.Module):
    """
    Neural network for learning covariance structure.

    Uses temporal patterns to estimate forward-looking covariance.
    """

    hidden_dim: int = 128
    num_assets: int = 10

    @nn.compact
    def __call__(
        self,
        returns: chex.Array,  # [batch, lookback, num_assets]
        train: bool = True
    ) -> chex.Array:
        """
        Estimate covariance matrix.

        Args:
            returns: Historical returns
            train: Training mode

        Returns:
            cov_matrix: Estimated covariance matrix [num_assets, num_assets]
        """
        # LSTM for temporal modeling
        lstm = nn.RNN(nn.LSTMCell(features=self.hidden_dim))

        # Process sequence
        carry = lstm.initialize_carry(jax.random.PRNGKey(0), returns.shape[:1] + (self.hidden_dim,))
        carry, outputs = lstm(carry, returns)

        # Final state
        final_state = outputs[-1]

        # Predict covariance parameters
        # Use Cholesky decomposition for valid covariance
        n_params = self.num_assets * (self.num_assets + 1) // 2
        chol_params = nn.Dense(n_params)(final_state)

        # Construct lower triangular matrix
        L = jnp.zeros((self.num_assets, self.num_assets))
        idx = jnp.tril_indices(self.num_assets)
        L = L.at[idx].set(chol_params)

        # Ensure positive diagonal (use softplus)
        L = L.at[jnp.diag_indices(self.num_assets)].set(
            nn.softplus(jnp.diag(L))
        )

        # Covariance matrix: Σ = L @ L^T
        cov_matrix = jnp.dot(L, L.T)

        return cov_matrix


class MeanVarianceOptimizer(nn.Module):
    """
    Differentiable mean-variance optimization layer.

    Implements Markowitz portfolio optimization with neural network.
    """

    num_assets: int = 10
    risk_aversion: float = 1.0

    @nn.compact
    def __call__(
        self,
        expected_returns: chex.Array,
        cov_matrix: chex.Array
    ) -> chex.Array:
        """
        Compute optimal portfolio weights.

        Args:
            expected_returns: Expected returns [num_assets]
            cov_matrix: Covariance matrix [num_assets, num_assets]

        Returns:
            weights: Optimal portfolio weights [num_assets]
        """
        # Quadratic programming: minimize w^T Σ w - λ * μ^T w
        # s.t. sum(w) = 1, w >= 0

        # Analytical solution (simplified, without constraints)
        # w = (1/λ) * Σ^{-1} * μ

        inv_cov = jnp.linalg.inv(cov_matrix + jnp.eye(self.num_assets) * 1e-6)
        weights = jnp.dot(inv_cov, expected_returns) / self.risk_aversion

        # Project to simplex (softmax approximation)
        weights = nn.softmax(weights * 10)  # Temperature scaling

        return weights


def create_lft_networks(
    state_dim: int,
    latent_dim: int,
    num_assets: int = 10,
    actor_hidden: Sequence[int] = (256, 256),
    critic_hidden: Sequence[int] = (256, 256),
    activation: str = "relu"
) -> LFTActorCritic:
    """
    Factory function to create LFT actor-critic networks.

    Args:
        state_dim: Dimension of state space
        latent_dim: Dimension of latent factors
        num_assets: Number of assets in portfolio
        actor_hidden: Hidden layer sizes for actor
        critic_hidden: Hidden layer sizes for critic
        activation: Activation function

    Returns:
        actor_critic: LFTActorCritic network
    """
    return LFTActorCritic(
        actor_hidden_dims=actor_hidden,
        critic_hidden_dims=critic_hidden,
        num_assets=num_assets,
        activation=activation
    )

"""
Market Data Generator for simulation and backtesting
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict
import chex


class MarketDataGenerator:
    """
    Generate synthetic market data for training and testing.

    Supports:
    - Geometric Brownian Motion (GBM)
    - Jump Diffusion
    - Heston Stochastic Volatility
    - Multi-asset with correlation
    - Order book simulation
    """

    def __init__(self, seed: int = 42):
        """Initialize data generator."""
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

    def generate_gbm(
        self,
        num_steps: int,
        num_assets: int,
        initial_price: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        dt: float = 1.0 / 252.0
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Generate price paths using Geometric Brownian Motion.

        dS = μ S dt + σ S dW

        Args:
            num_steps: Number of time steps
            num_assets: Number of assets
            initial_price: Initial price
            mu: Drift (annualized)
            sigma: Volatility (annualized)
            dt: Time step (e.g., 1/252 for daily)

        Returns:
            prices: Price paths [num_steps, num_assets, 1]
            returns: Log returns [num_steps, num_assets]
        """
        self.key, subkey = jax.random.split(self.key)

        # Generate random shocks
        shocks = jax.random.normal(subkey, (num_steps, num_assets))

        # Compute returns
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * jnp.sqrt(dt) * shocks
        returns = drift + diffusion

        # Compute prices
        log_prices = jnp.cumsum(returns, axis=0) + jnp.log(initial_price)
        prices = jnp.exp(log_prices)

        # Add OHLC structure (simplified: O=C_prev, H=C*1.02, L=C*0.98, C=C)
        ohlc_prices = jnp.zeros((num_steps, num_assets, 4))
        ohlc_prices = ohlc_prices.at[:, :, 3].set(prices)  # Close
        ohlc_prices = ohlc_prices.at[:, :, 0].set(
            jnp.concatenate([jnp.array([[initial_price] * num_assets]), prices[:-1]], axis=0)
        )  # Open
        ohlc_prices = ohlc_prices.at[:, :, 1].set(prices * 1.02)  # High
        ohlc_prices = ohlc_prices.at[:, :, 2].set(prices * 0.98)  # Low

        return ohlc_prices, returns

    def generate_jump_diffusion(
        self,
        num_steps: int,
        num_assets: int,
        initial_price: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        jump_intensity: float = 0.1,
        jump_size_mean: float = 0.0,
        jump_size_std: float = 0.05,
        dt: float = 1.0 / 252.0
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Generate prices with Merton Jump Diffusion model.

        dS = μ S dt + σ S dW + S dJ

        Args:
            jump_intensity: Poisson intensity (jumps per unit time)
            jump_size_mean: Mean jump size (log scale)
            jump_size_std: Jump size volatility

        Returns:
            prices: Price paths with jumps
            returns: Returns including jump components
        """
        # Generate base GBM
        ohlc_prices, gbm_returns = self.generate_gbm(
            num_steps, num_assets, initial_price, mu, sigma, dt
        )

        # Generate jumps
        self.key, key1, key2 = jax.random.split(self.key, 3)

        # Poisson process for jump times
        jump_mask = jax.random.poisson(key1, jump_intensity * dt, (num_steps, num_assets))
        jump_mask = (jump_mask > 0).astype(jnp.float32)

        # Jump sizes
        jump_sizes = jax.random.normal(key2, (num_steps, num_assets)) * jump_size_std + jump_size_mean

        # Add jumps to returns
        jump_returns = jump_mask * jump_sizes
        total_returns = gbm_returns + jump_returns

        # Recompute prices with jumps
        log_prices = jnp.cumsum(total_returns, axis=0) + jnp.log(initial_price)
        prices = jnp.exp(log_prices)

        # Update OHLC
        ohlc_prices = ohlc_prices.at[:, :, 3].set(prices)
        ohlc_prices = ohlc_prices.at[:, :, 1].set(jnp.maximum(ohlc_prices[:, :, 1], prices))
        ohlc_prices = ohlc_prices.at[:, :, 2].set(jnp.minimum(ohlc_prices[:, :, 2], prices))

        return ohlc_prices, total_returns

    def generate_heston(
        self,
        num_steps: int,
        num_assets: int,
        initial_price: float = 100.0,
        initial_vol: float = 0.2,
        kappa: float = 2.0,
        theta: float = 0.2,
        xi: float = 0.3,
        rho: float = -0.7,
        dt: float = 1.0 / 252.0
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Generate prices using Heston Stochastic Volatility model.

        dS = μ S dt + √v S dW1
        dv = κ(θ - v) dt + ξ √v dW2
        Corr(dW1, dW2) = ρ

        Args:
            kappa: Mean reversion speed
            theta: Long-term volatility
            xi: Volatility of volatility
            rho: Correlation between price and volatility

        Returns:
            prices: Price paths
            returns: Returns
            volatilities: Volatility paths
        """
        self.key, key1, key2 = jax.random.split(self.key, 3)

        # Generate correlated Brownian motions
        z1 = jax.random.normal(key1, (num_steps, num_assets))
        z2_indep = jax.random.normal(key2, (num_steps, num_assets))
        z2 = rho * z1 + jnp.sqrt(1 - rho ** 2) * z2_indep

        # Initialize arrays
        prices = jnp.zeros((num_steps, num_assets))
        volatilities = jnp.zeros((num_steps, num_assets))

        prices = prices.at[0].set(initial_price)
        volatilities = volatilities.at[0].set(initial_vol ** 2)

        # Euler-Maruyama discretization
        for t in range(1, num_steps):
            v_t = jnp.maximum(volatilities[t - 1], 1e-8)  # Floor volatility

            # Update volatility
            dv = kappa * (theta ** 2 - v_t) * dt + xi * jnp.sqrt(v_t) * jnp.sqrt(dt) * z2[t]
            v_next = jnp.maximum(v_t + dv, 1e-8)
            volatilities = volatilities.at[t].set(v_next)

            # Update price
            dS = jnp.sqrt(v_t) * jnp.sqrt(dt) * z1[t]
            S_next = prices[t - 1] * jnp.exp(dS)
            prices = prices.at[t].set(S_next)

        # Compute returns
        returns = jnp.diff(jnp.log(prices), axis=0, prepend=jnp.log(initial_price))

        # Create OHLC
        ohlc_prices = jnp.zeros((num_steps, num_assets, 4))
        ohlc_prices = ohlc_prices.at[:, :, 3].set(prices)
        ohlc_prices = ohlc_prices.at[:, :, 0].set(
            jnp.concatenate([jnp.array([[initial_price] * num_assets]), prices[:-1]], axis=0)
        )
        ohlc_prices = ohlc_prices.at[:, :, 1].set(prices * 1.02)
        ohlc_prices = ohlc_prices.at[:, :, 2].set(prices * 0.98)

        return ohlc_prices, returns, jnp.sqrt(volatilities)

    def generate_multi_asset_correlated(
        self,
        num_steps: int,
        num_assets: int,
        correlation_matrix: chex.Array,
        initial_prices: Optional[chex.Array] = None,
        mu: Optional[chex.Array] = None,
        sigma: Optional[chex.Array] = None,
        dt: float = 1.0 / 252.0
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Generate correlated multi-asset price paths.

        Args:
            correlation_matrix: Correlation matrix [num_assets, num_assets]
            initial_prices: Initial prices [num_assets]
            mu: Drift for each asset [num_assets]
            sigma: Volatility for each asset [num_assets]

        Returns:
            prices: Correlated price paths
            returns: Correlated returns
        """
        # Default parameters
        if initial_prices is None:
            initial_prices = jnp.ones(num_assets) * 100.0
        if mu is None:
            mu = jnp.ones(num_assets) * 0.05
        if sigma is None:
            sigma = jnp.ones(num_assets) * 0.2

        # Cholesky decomposition for correlation
        L = jnp.linalg.cholesky(correlation_matrix)

        # Generate independent standard normals
        self.key, subkey = jax.random.split(self.key)
        z = jax.random.normal(subkey, (num_steps, num_assets))

        # Apply correlation
        correlated_z = z @ L.T

        # Compute returns
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * jnp.sqrt(dt) * correlated_z
        returns = drift + diffusion

        # Compute prices
        log_prices = jnp.cumsum(returns, axis=0) + jnp.log(initial_prices)
        prices = jnp.exp(log_prices)

        # Create OHLC
        ohlc_prices = jnp.zeros((num_steps, num_assets, 4))
        ohlc_prices = ohlc_prices.at[:, :, 3].set(prices)
        ohlc_prices = ohlc_prices.at[:, :, 0].set(
            jnp.concatenate([initial_prices[None, :], prices[:-1]], axis=0)
        )
        ohlc_prices = ohlc_prices.at[:, :, 1].set(prices * 1.02)
        ohlc_prices = ohlc_prices.at[:, :, 2].set(prices * 0.98)

        return ohlc_prices, returns

    def generate_volume(
        self,
        num_steps: int,
        num_assets: int,
        base_volume: float = 1000000.0,
        volatility: float = 0.3
    ) -> chex.Array:
        """
        Generate trading volume data.

        Args:
            base_volume: Average daily volume
            volatility: Volume volatility

        Returns:
            volumes: Volume data [num_steps, num_assets]
        """
        self.key, subkey = jax.random.split(self.key)

        # Log-normal distribution for volumes
        log_vol = jnp.log(base_volume) + jax.random.normal(
            subkey, (num_steps, num_assets)
        ) * volatility

        volumes = jnp.exp(log_vol)

        return volumes

    def generate_orderbook_data(
        self,
        num_ticks: int,
        depth_levels: int = 10,
        initial_price: float = 100.0,
        tick_size: float = 0.01,
        volatility: float = 0.0001
    ) -> Dict[str, chex.Array]:
        """
        Generate synthetic order book data.

        Returns:
            Dictionary with:
            - bid_prices: [num_ticks, depth_levels]
            - bid_volumes: [num_ticks, depth_levels]
            - ask_prices: [num_ticks, depth_levels]
            - ask_volumes: [num_ticks, depth_levels]
            - mid_prices: [num_ticks]
        """
        self.key, key1, key2, key3, key4 = jax.random.split(self.key, 5)

        # Generate mid price path (random walk)
        mid_price_changes = jax.random.normal(key1, (num_ticks,)) * volatility
        mid_prices = initial_price + jnp.cumsum(mid_price_changes)

        # Generate bid side
        bid_offsets = jnp.arange(1, depth_levels + 1) * tick_size
        bid_prices = mid_prices[:, None] - bid_offsets[None, :]

        bid_volumes = jax.random.lognormal(key2, shape=(num_ticks, depth_levels)) * 1000

        # Generate ask side
        ask_offsets = jnp.arange(1, depth_levels + 1) * tick_size
        ask_prices = mid_prices[:, None] + ask_offsets[None, :]

        ask_volumes = jax.random.lognormal(key3, shape=(num_ticks, depth_levels)) * 1000

        return {
            'bid_prices': bid_prices,
            'bid_volumes': bid_volumes,
            'ask_prices': ask_prices,
            'ask_volumes': ask_volumes,
            'mid_prices': mid_prices
        }

    def generate_macro_features(
        self,
        num_steps: int,
        num_features: int = 10
    ) -> chex.Array:
        """
        Generate macro economic features.

        Features may include:
        - VIX (volatility index)
        - Interest rates
        - Market regime indicators
        - Liquidity measures

        Returns:
            features: Macro features [num_steps, num_features]
        """
        self.key, subkey = jax.random.split(self.key)

        # Generate features with autocorrelation
        features = []

        for _ in range(num_features):
            self.key, subkey = jax.random.split(self.key)

            # AR(1) process: x_t = φ x_{t-1} + ε_t
            phi = 0.95  # High autocorrelation
            innovations = jax.random.normal(subkey, (num_steps,))

            feature_values = jnp.zeros(num_steps)
            for t in range(1, num_steps):
                feature_values = feature_values.at[t].set(
                    phi * feature_values[t - 1] + innovations[t]
                )

            features.append(feature_values)

        features = jnp.stack(features, axis=1)

        # Standardize
        features = (features - jnp.mean(features, axis=0)) / (jnp.std(features, axis=0) + 1e-8)

        return features

    @staticmethod
    def create_correlation_matrix(
        num_assets: int,
        block_correlation: float = 0.5,
        num_blocks: int = 2
    ) -> chex.Array:
        """
        Create a block-diagonal correlation matrix.

        Args:
            num_assets: Number of assets
            block_correlation: Within-block correlation
            num_blocks: Number of correlation blocks

        Returns:
            correlation_matrix: [num_assets, num_assets]
        """
        # Start with identity
        corr_matrix = jnp.eye(num_assets)

        # Add block structure
        assets_per_block = num_assets // num_blocks

        for block_idx in range(num_blocks):
            start_idx = block_idx * assets_per_block
            end_idx = start_idx + assets_per_block if block_idx < num_blocks - 1 else num_assets

            # Set within-block correlation
            for i in range(start_idx, end_idx):
                for j in range(start_idx, end_idx):
                    if i != j:
                        corr_matrix = corr_matrix.at[i, j].set(block_correlation)

        return corr_matrix

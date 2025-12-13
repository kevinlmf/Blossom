"""
Factor Extractor for interpretable factor analysis and extraction.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, List, Tuple, Optional
import chex


class FactorExtractor:
    """
    Utility class for extracting and analyzing latent factors.

    Provides:
    - Factor decomposition
    - Factor attribution
    - Factor timing analysis
    - Cross-sectional factor analysis
    """

    def __init__(
        self,
        num_factors: int,
        factor_names: Optional[List[str]] = None
    ):
        """
        Initialize factor extractor.

        Args:
            num_factors: Number of latent factors
            factor_names: Optional names for factors
        """
        self.num_factors = num_factors
        self.factor_names = factor_names or [f"Factor_{i+1}" for i in range(num_factors)]

    def extract_factor_returns(
        self,
        latent_factors: chex.Array,
        asset_returns: chex.Array,
        method: str = "fama_macbeth"
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Extract factor returns using cross-sectional regression.

        Args:
            latent_factors: Learned factors [T, num_factors]
            asset_returns: Asset returns [T, num_assets]
            method: Extraction method ('fama_macbeth', 'ols')

        Returns:
            factor_returns: Factor returns [T, num_factors]
            factor_premiums: Average factor premiums [num_factors]
        """
        if method == "fama_macbeth":
            return self._fama_macbeth_regression(latent_factors, asset_returns)
        elif method == "ols":
            return self._ols_regression(latent_factors, asset_returns)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _fama_macbeth_regression(
        self,
        factors: chex.Array,
        returns: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Fama-MacBeth two-pass regression.

        Step 1: Time-series regression to estimate betas
        Step 2: Cross-sectional regression to estimate factor premiums
        """
        T, K = factors.shape  # Time, Num factors
        T, N = returns.shape  # Time, Num assets

        # Step 1: Estimate betas (time-series regression)
        # For each asset: R_i = alpha_i + beta_i * F + epsilon_i
        F_with_intercept = jnp.concatenate([jnp.ones((T, 1)), factors], axis=1)
        betas = jnp.linalg.lstsq(F_with_intercept, returns, rcond=None)[0]
        betas = betas[1:, :]  # Remove intercept [K, N]

        # Step 2: Cross-sectional regression for each time period
        # For each time t: R_t = gamma_0 + gamma * beta + epsilon_t
        factor_returns = []

        for t in range(T):
            R_t = returns[t, :]  # [N]
            beta_with_intercept = jnp.concatenate([
                jnp.ones((1, N)), betas
            ], axis=0).T  # [N, K+1]

            # Cross-sectional regression
            gamma = jnp.linalg.lstsq(beta_with_intercept, R_t, rcond=None)[0]
            factor_returns.append(gamma[1:])  # Exclude intercept

        factor_returns = jnp.array(factor_returns)  # [T, K]
        factor_premiums = jnp.mean(factor_returns, axis=0)  # [K]

        return factor_returns, factor_premiums

    def _ols_regression(
        self,
        factors: chex.Array,
        returns: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Simple OLS regression for factor returns."""
        # Assume factors are already returns-like
        factor_premiums = jnp.mean(factors, axis=0)
        return factors, factor_premiums

    def compute_factor_statistics(
        self,
        factor_returns: chex.Array,
        risk_free_rate: float = 0.0
    ) -> Dict[str, chex.Array]:
        """
        Compute comprehensive factor statistics.

        Args:
            factor_returns: Factor returns [T, num_factors]
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Dictionary of factor statistics
        """
        T, K = factor_returns.shape

        # Mean returns
        mean_returns = jnp.mean(factor_returns, axis=0)

        # Volatility
        volatility = jnp.std(factor_returns, axis=0)

        # Sharpe ratio
        excess_returns = factor_returns - risk_free_rate / 252
        sharpe_ratio = jnp.mean(excess_returns, axis=0) / (jnp.std(excess_returns, axis=0) + 1e-8)

        # Maximum drawdown
        cumulative_returns = jnp.cumprod(1 + factor_returns, axis=0)
        running_max = jnp.maximum.accumulate(cumulative_returns, axis=0)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = jnp.min(drawdown, axis=0)

        # Correlation matrix
        correlation = jnp.corrcoef(factor_returns.T)

        # Skewness
        centered = factor_returns - mean_returns
        skewness = jnp.mean(centered ** 3, axis=0) / (volatility ** 3 + 1e-8)

        # Kurtosis
        kurtosis = jnp.mean(centered ** 4, axis=0) / (volatility ** 4 + 1e-8) - 3

        return {
            'mean_returns': mean_returns,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'correlation': correlation,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def compute_factor_attribution(
        self,
        portfolio_returns: chex.Array,
        factor_returns: chex.Array,
        factor_exposures: chex.Array
    ) -> Dict[str, chex.Array]:
        """
        Decompose portfolio returns into factor contributions.

        Args:
            portfolio_returns: Portfolio returns [T]
            factor_returns: Factor returns [T, num_factors]
            factor_exposures: Factor exposures (betas) [num_factors]

        Returns:
            Dictionary with factor attributions
        """
        # Factor contribution = beta * factor_return
        factor_contributions = factor_exposures[None, :] * factor_returns  # [T, K]

        # Residual returns
        total_factor_return = jnp.sum(factor_contributions, axis=1)
        residual_returns = portfolio_returns - total_factor_return

        # Contribution to total return
        total_return = jnp.sum(portfolio_returns)
        factor_pct_contribution = jnp.sum(factor_contributions, axis=0) / total_return

        return {
            'factor_contributions': factor_contributions,
            'residual_returns': residual_returns,
            'factor_pct_contribution': factor_pct_contribution,
            'total_factor_return': total_factor_return
        }

    def detect_factor_timing(
        self,
        factor_returns: chex.Array,
        window: int = 60
    ) -> Dict[str, chex.Array]:
        """
        Detect factor timing signals using rolling statistics.

        Args:
            factor_returns: Factor returns [T, num_factors]
            window: Rolling window size

        Returns:
            Dictionary with timing signals
        """
        T, K = factor_returns.shape

        # Rolling mean (momentum signal)
        rolling_mean = jnp.array([
            jnp.convolve(factor_returns[:, k], jnp.ones(window)/window, mode='valid')
            for k in range(K)
        ]).T

        # Rolling volatility
        rolling_vol = jnp.array([
            jnp.sqrt(jnp.convolve(factor_returns[:, k]**2, jnp.ones(window)/window, mode='valid'))
            for k in range(K)
        ]).T

        # Momentum signal
        momentum_signal = rolling_mean / (rolling_vol + 1e-8)

        # Trend signal (positive momentum)
        trend_signal = (rolling_mean > 0).astype(jnp.float32)

        # Volatility regime (high vol = 1, low vol = 0)
        median_vol = jnp.median(rolling_vol, axis=0, keepdims=True)
        vol_regime = (rolling_vol > median_vol).astype(jnp.float32)

        return {
            'momentum_signal': momentum_signal,
            'trend_signal': trend_signal,
            'vol_regime': vol_regime,
            'rolling_mean': rolling_mean,
            'rolling_vol': rolling_vol
        }

    def compute_factor_exposures(
        self,
        portfolio_weights: chex.Array,
        asset_betas: chex.Array
    ) -> chex.Array:
        """
        Compute portfolio-level factor exposures.

        Args:
            portfolio_weights: Portfolio weights [num_assets]
            asset_betas: Asset factor betas [num_assets, num_factors]

        Returns:
            portfolio_betas: Portfolio factor exposures [num_factors]
        """
        # Portfolio beta = weighted average of asset betas
        portfolio_betas = portfolio_weights @ asset_betas
        return portfolio_betas

    def orthogonalize_factors(
        self,
        factors: chex.Array,
        method: str = "gram_schmidt"
    ) -> chex.Array:
        """
        Orthogonalize factors to remove multicollinearity.

        Args:
            factors: Input factors [T, num_factors]
            method: Orthogonalization method

        Returns:
            orthogonal_factors: Orthogonalized factors [T, num_factors]
        """
        if method == "gram_schmidt":
            return self._gram_schmidt_orthogonalization(factors)
        elif method == "pca":
            return self._pca_orthogonalization(factors)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _gram_schmidt_orthogonalization(
        self,
        factors: chex.Array
    ) -> chex.Array:
        """Gram-Schmidt orthogonalization."""
        T, K = factors.shape
        orthogonal_factors = jnp.zeros_like(factors)

        for k in range(K):
            # Start with original factor
            v = factors[:, k]

            # Subtract projections on previous orthogonal factors
            for j in range(k):
                u = orthogonal_factors[:, j]
                projection = jnp.dot(v, u) / (jnp.dot(u, u) + 1e-8)
                v = v - projection * u

            # Normalize
            orthogonal_factors = orthogonal_factors.at[:, k].set(
                v / (jnp.linalg.norm(v) + 1e-8)
            )

        return orthogonal_factors

    def _pca_orthogonalization(
        self,
        factors: chex.Array
    ) -> chex.Array:
        """PCA-based orthogonalization."""
        # Standardize factors
        factors_std = (factors - jnp.mean(factors, axis=0)) / (jnp.std(factors, axis=0) + 1e-8)

        # Compute covariance matrix
        cov_matrix = jnp.cov(factors_std.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = jnp.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Transform factors
        orthogonal_factors = factors_std @ eigenvectors

        return orthogonal_factors

    def get_factor_names(self) -> List[str]:
        """Get factor names."""
        return self.factor_names

    def set_factor_names(self, names: List[str]) -> None:
        """Set factor names."""
        if len(names) != self.num_factors:
            raise ValueError(f"Expected {self.num_factors} names, got {len(names)}")
        self.factor_names = names

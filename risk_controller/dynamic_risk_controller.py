"""
Dynamic Risk Controller (Replacement for Static Copula)

More appropriate for multi-frequency trading systems:
1. DCC-GARCH for time-varying correlations
2. Rolling window risk metrics
3. Regime-aware risk management
4. Real-time CVaR with exponential weighting
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
from collections import deque


class DynamicRiskController:
    """
    Dynamic Risk Controller for Multi-Frequency Trading

    Key improvements over static Copula:
    1. Time-varying correlation using exponential moving average
    2. Regime-dependent risk thresholds
    3. Frequency-specific risk metrics
    4. Real-time adaptable CVaR
    """

    def __init__(
        self,
        num_agents: int = 3,
        ema_alpha: float = 0.94,  # Exponential smoothing for correlation
        cvar_alpha: float = 0.05,  # CVaR confidence level
        lookback_window: int = 100,
        regime_threshold: float = 0.02  # Volatility threshold for regime detection
    ):
        """
        Args:
            num_agents: Number of agents (HFT, MFT, LFT)
            ema_alpha: Smoothing parameter for exponential moving average
            cvar_alpha: Confidence level for CVaR
            lookback_window: Window for risk metrics
            regime_threshold: Threshold to detect high volatility regime
        """
        self.num_agents = num_agents
        self.ema_alpha = ema_alpha
        self.cvar_alpha = cvar_alpha
        self.lookback_window = lookback_window
        self.regime_threshold = regime_threshold

        # Time-varying correlation matrix (exponentially weighted)
        self.correlation_matrix = jnp.eye(num_agents)

        # Exponentially weighted covariance matrix
        self.cov_matrix = jnp.eye(num_agents) * 0.01

        # Rolling returns for each agent
        self.returns_buffers = {
            'hft': deque(maxlen=lookback_window),
            'mft': deque(maxlen=lookback_window),
            'lft': deque(maxlen=lookback_window)
        }

        # Exponentially weighted returns (for fast adaptation)
        self.ema_returns = jnp.zeros(num_agents)
        self.ema_volatility = jnp.ones(num_agents) * 0.01

        # Market regime
        self.current_regime = "normal"  # "normal", "high_vol", "crisis"

        # Step counter
        self.step_count = 0

        # Goal-aligned constraint overrides
        self.goal_constraints: Dict[str, float] = {}

    def update(
        self,
        hft_return: float,
        mft_return: float,
        lft_return: float
    ):
        """
        Update risk controller with new returns

        Uses exponential weighting for fast adaptation to regime changes
        """
        # Guard against NaN/inf
        returns = jnp.array([hft_return, mft_return, lft_return])
        returns = jnp.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        returns = jnp.clip(returns, -1.0, 1.0)

        # Update buffers
        self.returns_buffers['hft'].append(float(returns[0]))
        self.returns_buffers['mft'].append(float(returns[1]))
        self.returns_buffers['lft'].append(float(returns[2]))

        # Update exponentially weighted mean returns
        self.ema_returns = self.ema_alpha * self.ema_returns + (1 - self.ema_alpha) * returns

        # Update exponentially weighted volatility
        squared_deviations = (returns - self.ema_returns) ** 2
        self.ema_volatility = jnp.sqrt(
            self.ema_alpha * self.ema_volatility**2 + (1 - self.ema_alpha) * squared_deviations
        )

        # Update covariance matrix using exponential weighting
        if self.step_count >= 10:
            self._update_dynamic_correlation(returns)

        # Update market regime
        self._update_regime()

        self.step_count += 1

    def _update_dynamic_correlation(self, returns: jnp.ndarray):
        """
        Update correlation matrix using exponentially weighted moving average

        This is much more responsive to regime changes than static Copula
        """
        # Standardized returns
        std_returns = (returns - self.ema_returns) / (self.ema_volatility + 1e-8)
        std_returns = jnp.nan_to_num(std_returns, nan=0.0)

        # Update covariance using exponential smoothing
        outer_product = jnp.outer(std_returns, std_returns)
        self.cov_matrix = self.ema_alpha * self.cov_matrix + (1 - self.ema_alpha) * outer_product

        # Convert to correlation
        std_devs = jnp.sqrt(jnp.diag(self.cov_matrix))
        std_devs = jnp.maximum(std_devs, 1e-8)

        self.correlation_matrix = self.cov_matrix / jnp.outer(std_devs, std_devs)

        # Ensure valid correlation matrix
        self.correlation_matrix = jnp.nan_to_num(self.correlation_matrix, nan=0.0)
        self.correlation_matrix = jnp.clip(self.correlation_matrix, -1.0, 1.0)
        self.correlation_matrix = self.correlation_matrix.at[jnp.diag_indices(self.num_agents)].set(1.0)

    def _update_regime(self):
        """
        Detect market regime based on portfolio volatility

        Regimes:
        - normal: low volatility
        - high_vol: elevated volatility
        - crisis: extreme volatility
        """
        portfolio_vol = jnp.mean(self.ema_volatility)

        if portfolio_vol < self.regime_threshold:
            self.current_regime = "normal"
        elif portfolio_vol < 2 * self.regime_threshold:
            self.current_regime = "high_vol"
        else:
            self.current_regime = "crisis"

    def get_risk_penalty(self) -> float:
        """
        Calculate risk penalty based on:
        1. Correlation among agents (systemic risk)
        2. Current market regime
        3. Concentration risk

        Returns:
            Risk penalty (higher = more risk)
        """
        if self.step_count < 10:
            return 0.0

        # 1. Correlation penalty (systemic risk)
        n = self.num_agents
        off_diagonal_sum = jnp.sum(jnp.abs(self.correlation_matrix)) - n
        avg_correlation = off_diagonal_sum / (n * (n - 1))
        correlation_penalty = avg_correlation

        # 2. Regime adjustment
        regime_multipliers = {
            "normal": 1.0,
            "high_vol": 1.5,
            "crisis": 2.5
        }
        regime_multiplier = regime_multipliers[self.current_regime]

        # 3. Volatility penalty
        volatility_penalty = jnp.mean(self.ema_volatility) / self.regime_threshold

        # Combined penalty
        total_penalty = (0.5 * correlation_penalty + 0.5 * volatility_penalty) * regime_multiplier

        # Goal constraint penalties
        if self.goal_constraints:
            if 'max_drawdown' in self.goal_constraints and self.step_count >= 20:
                allowed_drawdown = abs(float(self.goal_constraints['max_drawdown']))
                current_drawdown = self.get_drawdown_risk()
                if current_drawdown > allowed_drawdown:
                    total_penalty += float(current_drawdown - allowed_drawdown)

            if 'max_cvar_95' in self.goal_constraints and self.step_count >= 20:
                allowed_cvar = abs(float(self.goal_constraints['max_cvar_95']))
                current_cvar = abs(self.get_joint_cvar())
                if current_cvar > allowed_cvar:
                    total_penalty += float((current_cvar - allowed_cvar) * 10.0)

        return float(jnp.clip(total_penalty, 0.0, 5.0))

    def get_joint_cvar(self, alpha: float = None) -> float:
        """
        Calculate joint CVaR using exponentially weighted returns

        More responsive than static Copula approach
        """
        if alpha is None:
            alpha = self.cvar_alpha

        if len(self.returns_buffers['hft']) < 20:
            return 0.0

        # Get returns
        hft_returns = jnp.array(list(self.returns_buffers['hft']))
        mft_returns = jnp.array(list(self.returns_buffers['mft']))
        lft_returns = jnp.array(list(self.returns_buffers['lft']))

        # Guard against NaN
        hft_returns = jnp.nan_to_num(hft_returns, nan=0.0)
        mft_returns = jnp.nan_to_num(mft_returns, nan=0.0)
        lft_returns = jnp.nan_to_num(lft_returns, nan=0.0)

        # Equal weight portfolio
        portfolio_returns = (hft_returns + mft_returns + lft_returns) / 3.0

        # Apply exponential weighting (recent returns matter more)
        n = len(portfolio_returns)
        weights = jnp.array([self.ema_alpha ** (n - i - 1) for i in range(n)])
        weights = weights / jnp.sum(weights)

        # Weighted VaR
        sorted_indices = jnp.argsort(portfolio_returns)
        sorted_returns = portfolio_returns[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum_weights = jnp.cumsum(sorted_weights)
        var_idx = jnp.searchsorted(cumsum_weights, alpha)
        var_threshold = sorted_returns[var_idx] if var_idx < len(sorted_returns) else sorted_returns[-1]

        # Weighted CVaR
        tail_mask = portfolio_returns <= var_threshold
        if jnp.sum(tail_mask * weights) > 0:
            cvar = jnp.sum(portfolio_returns * tail_mask * weights) / jnp.sum(tail_mask * weights)
        else:
            cvar = var_threshold

        cvar = jnp.nan_to_num(cvar, nan=0.0)
        cvar = jnp.clip(cvar, -10.0, 0.0)

        return float(cvar)

    def get_concentration_risk(self) -> float:
        """
        Measure concentration risk across agents

        Returns:
            Concentration score (0 = diversified, 1 = concentrated)
        """
        if self.step_count < 10:
            return 0.0

        # Use recent returns to measure concentration
        recent_returns = jnp.array([
            jnp.mean(jnp.array(list(self.returns_buffers['hft'])[-10:])),
            jnp.mean(jnp.array(list(self.returns_buffers['mft'])[-10:])),
            jnp.mean(jnp.array(list(self.returns_buffers['lft'])[-10:]))
        ])

        recent_returns = jnp.nan_to_num(recent_returns, nan=0.0)

        # Herfindahl index (normalized)
        abs_returns = jnp.abs(recent_returns) + 1e-8
        proportions = abs_returns / jnp.sum(abs_returns)
        herfindahl = jnp.sum(proportions ** 2)

        # Normalize: 1/n (diversified) to 1 (concentrated)
        normalized_concentration = (herfindahl - 1/self.num_agents) / (1 - 1/self.num_agents)

        return float(jnp.clip(normalized_concentration, 0.0, 1.0))

    def get_drawdown_risk(self) -> float:
        """
        Calculate drawdown risk across agents
        """
        if self.step_count < 20:
            return 0.0

        drawdowns = []
        for agent in ['hft', 'mft', 'lft']:
            returns = jnp.array(list(self.returns_buffers[agent]))
            returns = jnp.nan_to_num(returns, nan=0.0)

            cumulative_returns = jnp.cumsum(returns)
            running_max = jnp.maximum.accumulate(cumulative_returns)
            drawdown = (running_max - cumulative_returns) / (jnp.abs(running_max) + 1e-8)

            max_drawdown = jnp.max(drawdown)
            drawdowns.append(float(max_drawdown))

        avg_drawdown = jnp.mean(jnp.array(drawdowns))
        return float(jnp.clip(avg_drawdown, 0.0, 1.0))

    def get_regime_info(self) -> Dict[str, any]:
        """Get current market regime information"""
        return {
            'regime': self.current_regime,
            'portfolio_volatility': float(jnp.mean(self.ema_volatility)),
            'correlation_matrix': self.correlation_matrix.tolist(),
            'ema_returns': self.ema_returns.tolist()
        }

    def get_risk_report(self) -> Dict[str, any]:
        """
        Comprehensive risk report
        """
        return {
            'risk_penalty': self.get_risk_penalty(),
            'joint_cvar': self.get_joint_cvar(),
            'concentration_risk': self.get_concentration_risk(),
            'drawdown_risk': self.get_drawdown_risk(),
            'regime': self.current_regime,
            'correlation_matrix': self.correlation_matrix.tolist(),
            'agent_volatilities': self.ema_volatility.tolist(),
            'num_observations': self.step_count,
            'goal_constraints': dict(self.goal_constraints)
        }

    def reset(self):
        """Reset risk controller state"""
        self.correlation_matrix = jnp.eye(self.num_agents)
        self.cov_matrix = jnp.eye(self.num_agents) * 0.01
        self.returns_buffers = {
            'hft': deque(maxlen=self.lookback_window),
            'mft': deque(maxlen=self.lookback_window),
            'lft': deque(maxlen=self.lookback_window)
        }
        self.ema_returns = jnp.zeros(self.num_agents)
        self.ema_volatility = jnp.ones(self.num_agents) * 0.01
        self.current_regime = "normal"
        self.step_count = 0
        self.goal_constraints = {}

    def set_goal_constraints(self, constraints: Dict[str, float]):
        """
        Apply goal-driven constraint overrides (e.g., max drawdown, CVaR).
        """
        self.goal_constraints = dict(constraints)


# Alias for backward compatibility
RiskController = DynamicRiskController

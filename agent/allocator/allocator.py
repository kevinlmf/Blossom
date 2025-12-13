"""
Capital Allocator

Meta-level agent that allocates capital across HFT, MFT, and LFT agents
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional
import flax.linen as nn


class CapitalAllocator:
    """
    Capital Allocator Agent (Meta-Level)

    Allocates capital across HFT/MFT/LFT agents to maximize:
    - Long-term wealth
    - Risk-adjusted returns (Sharpe ratio)
    - CVaR minimization

    Uses reinforcement learning to learn optimal allocation policy
    """

    def __init__(
        self,
        num_agents: int = 3,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        sharpe_weight: float = 0.3,
        cvar_weight: float = 0.3,
        wealth_weight: float = 0.4,
        lookback_window: int = 50,
        enable_adaptive_weights: bool = True,
        reallocation_cost: float = 0.001
    ):
        """
        Args:
            num_agents: Number of sub-agents to allocate to (HFT, MFT, LFT)
            hidden_dims: Hidden layer dimensions for allocation network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            sharpe_weight: Weight for Sharpe ratio in reward (default, will be adaptive)
            cvar_weight: Weight for CVaR in reward (default, will be adaptive)
            wealth_weight: Weight for wealth in reward (default, will be adaptive)
            lookback_window: Historical window for performance metrics
            enable_adaptive_weights: Enable market-adaptive reward weights
            reallocation_cost: Transaction cost for changing allocation (as % of notional)
        """
        self.num_agents = num_agents
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Default weights (used when adaptive is disabled or as fallback)
        self.default_sharpe_weight = sharpe_weight
        self.default_cvar_weight = cvar_weight
        self.default_wealth_weight = wealth_weight

        # Current adaptive weights
        self.sharpe_weight = sharpe_weight
        self.cvar_weight = cvar_weight
        self.wealth_weight = wealth_weight

        self.lookback_window = lookback_window
        self.enable_adaptive_weights = enable_adaptive_weights
        self.reallocation_cost = reallocation_cost

        # Performance tracking
        self.agent_performance_history = {
            'hft': [],
            'mft': [],
            'lft': []
        }

        self.allocation_history = []
        self.wealth_history = [1.0]  # Start with normalized wealth

        # Market regime tracking
        self.current_market_regime = 'normal'
        self.regime_history = []

        # Guardrail configuration
        self.goal_allocation_targets: Optional[Dict[str, float]] = None
        self.goal_risk_targets: Dict[str, float] = {}
        self.goal_alignment_strength: float = 0.3
        self.max_step_change = 0.25  # Max change per allocation step
        self.min_allocation_floor = 0.05
        self.guardrail_trigger_count = 0
        self.last_allocation: Optional[jnp.ndarray] = None
        self.fallback_allocation = jnp.ones(self.num_agents) / self.num_agents

    def allocate(
        self,
        state: Dict[str, jnp.ndarray],
        agent_performances: Dict[str, float],
        latent_factors: jnp.ndarray,
        macro_indicators: Dict[str, float]
    ) -> jnp.ndarray:
        """
        Allocate capital across agents

        Args:
            state: Current system state
            agent_performances: Recent performance of each agent
            latent_factors: Latent factors from shared encoder
            macro_indicators: Macro market indicators

        Returns:
            allocation: Capital allocation ratios [π_HFT, π_MFT, π_LFT]
        """
        # Update performance history
        for agent in ['hft', 'mft', 'lft']:
            perf = agent_performances.get(agent, 0.0)
            self.agent_performance_history[agent].append(perf)
            if len(self.agent_performance_history[agent]) > self.lookback_window:
                self.agent_performance_history[agent].pop(0)

        # Calculate allocation features
        features = self._extract_allocation_features(
            agent_performances, latent_factors, macro_indicators
        )

        # Simple allocation strategy (to be replaced with learned policy)
        allocation = self._simple_allocation_strategy(features)

        # Apply goal alignment blend
        if self.goal_allocation_targets:
            goal_vector = jnp.array([
                self.goal_allocation_targets.get('hft', 1/3),
                self.goal_allocation_targets.get('mft', 1/3),
                self.goal_allocation_targets.get('lft', 1/3)
            ])
            goal_vector = self._normalize_allocation(goal_vector)
            blend = jnp.clip(self.goal_alignment_strength, 0.0, 1.0)
            allocation = (1.0 - blend) * allocation + blend * goal_vector
            allocation = self._normalize_allocation(allocation)

        allocation = self._apply_guardrails(allocation, macro_indicators)

        self.allocation_history.append(allocation)
        self.last_allocation = allocation

        return allocation

    def _extract_allocation_features(
        self,
        agent_performances: Dict[str, float],
        latent_factors: jnp.ndarray,
        macro_indicators: Dict[str, float]
    ) -> jnp.ndarray:
        """
        Extract features for allocation decision

        Returns:
            Feature vector for allocation network
        """
        features = []

        # Agent performance features
        for agent in ['hft', 'mft', 'lft']:
            if len(self.agent_performance_history[agent]) > 0:
                history = jnp.array(self.agent_performance_history[agent])

                # Mean return
                mean_return = jnp.mean(history)
                features.append(mean_return)

                # Volatility
                volatility = jnp.std(history)
                features.append(volatility)

                # Sharpe ratio
                sharpe = mean_return / (volatility + 1e-8)
                features.append(sharpe)

                # Recent performance (last 5 periods)
                recent_perf = jnp.mean(history[-5:]) if len(history) >= 5 else mean_return
                features.append(recent_perf)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        # Latent factors
        features.extend(latent_factors.tolist())

        # Macro indicators
        features.append(macro_indicators.get('volatility', 0.0))
        features.append(macro_indicators.get('liquidity', 0.0))
        features.append(macro_indicators.get('regime', 0.0))

        return jnp.array(features)

    def _simple_allocation_strategy(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Simple allocation strategy based on Sharpe ratios

        This is a placeholder - should be replaced with learned policy

        Args:
            features: Feature vector

        Returns:
            allocation: Capital allocation ratios
        """
        # Extract Sharpe ratios from features
        # Features structure: [hft_mean, hft_vol, hft_sharpe, hft_recent, ...]
        hft_sharpe = features[2]
        mft_sharpe = features[6]
        lft_sharpe = features[10]

        # Guard against NaN/inf
        hft_sharpe = jnp.nan_to_num(hft_sharpe, nan=0.0, posinf=0.0, neginf=0.0)
        mft_sharpe = jnp.nan_to_num(mft_sharpe, nan=0.0, posinf=0.0, neginf=0.0)
        lft_sharpe = jnp.nan_to_num(lft_sharpe, nan=0.0, posinf=0.0, neginf=0.0)

        # Simple strategy: allocate proportional to Sharpe ratio
        sharpe_ratios = jnp.array([hft_sharpe, mft_sharpe, lft_sharpe])

        # Clip negative Sharpe ratios to zero
        sharpe_ratios = jnp.maximum(sharpe_ratios, 0.0)

        # Add minimum allocation constraint (at least 5% per agent)
        min_allocation = 0.05

        # Allocate proportionally
        if jnp.sum(sharpe_ratios) > 1e-8:
            base_allocation = sharpe_ratios / jnp.sum(sharpe_ratios)
            # Ensure minimum allocation
            allocation = base_allocation * (1 - self.num_agents * min_allocation) + min_allocation
        else:
            # Equal allocation if all Sharpe ratios are zero/NaN
            allocation = jnp.ones(self.num_agents) / self.num_agents

        return allocation

    def _apply_guardrails(self, allocation: jnp.ndarray, macro: Dict[str, float]) -> jnp.ndarray:
        """Clip allocations, limit step change, and fallback on breaches."""
        allocation = self._normalize_allocation(allocation)

        # Minimum allocation floor from goal directive if available
        floor = max(self.min_allocation_floor, self.goal_risk_targets.get('allocation_floor', 0.0))
        allocation = jnp.maximum(allocation, floor)
        allocation = self._normalize_allocation(allocation)

        if self.last_allocation is not None:
            change = allocation - self.last_allocation
            clipped_change = jnp.clip(change, -self.max_step_change, self.max_step_change)
            allocation = self.last_allocation + clipped_change
            allocation = self._normalize_allocation(allocation)

        # Risk trigger based on macro volatility vs targets
        volatility = float(macro.get('volatility', 0.0))
        risk_aversion_target = float(self.goal_risk_targets.get('risk_aversion', 1.0))
        if volatility > risk_aversion_target * 1.5:
            # Trigger fallback to more defensive allocation
            self.guardrail_trigger_count += 1
            defensive = jnp.array([
                0.2,
                0.3,
                0.5,
            ])
            defensive = self._normalize_allocation(defensive)
            allocation = 0.5 * allocation + 0.5 * defensive
            allocation = self._normalize_allocation(allocation)

        return allocation

    def _normalize_allocation(self, allocation: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize allocation to sum to 1 and ensure non-negative
        """
        # Clip to non-negative
        allocation = jnp.maximum(allocation, 0.0)

        # Normalize to sum to 1
        total = jnp.sum(allocation)
        if total > 0:
            allocation = allocation / total
        else:
            # Equal allocation if all values are zero
            allocation = jnp.ones(self.num_agents) / self.num_agents

        return allocation

    def _detect_market_regime(
        self,
        macro_indicators: Dict[str, float]
    ) -> str:
        """
        Detect current market regime based on macro indicators

        Regimes:
        - 'trending_bull': High returns, moderate volatility, strong trend
        - 'trending_bear': Negative returns, high volatility, strong downtrend
        - 'high_volatility': High volatility regardless of direction (crisis/uncertainty)
        - 'low_volatility': Low volatility, range-bound market
        - 'low_liquidity': Low liquidity conditions
        - 'normal': Normal market conditions

        Args:
            macro_indicators: Dictionary with 'volatility', 'liquidity', 'regime'

        Returns:
            Market regime string
        """
        volatility = float(macro_indicators.get('volatility', 0.0))
        liquidity = float(macro_indicators.get('liquidity', 1.0))
        regime = float(macro_indicators.get('regime', 0.0))

        # Thresholds (can be tuned)
        high_vol_threshold = 0.3
        low_vol_threshold = 0.1
        low_liquidity_threshold = 0.5
        bull_threshold = 0.2
        bear_threshold = -0.2

        # Priority logic: more severe conditions first
        if liquidity < low_liquidity_threshold:
            return 'low_liquidity'
        elif volatility > high_vol_threshold:
            if regime < bear_threshold:
                return 'trending_bear'
            else:
                return 'high_volatility'
        elif volatility < low_vol_threshold:
            return 'low_volatility'
        elif regime > bull_threshold:
            return 'trending_bull'
        elif regime < bear_threshold:
            return 'trending_bear'
        else:
            return 'normal'

    def _get_adaptive_weights(
        self,
        market_regime: str
    ) -> Tuple[float, float, float]:
        """
        Get adaptive reward weights based on market regime

        Strategy:
        - trending_bull: Aggressive (prioritize wealth growth)
        - trending_bear: Defensive (prioritize CVaR protection)
        - high_volatility: Risk control (high CVaR weight)
        - low_volatility: Balanced (favor Sharpe for stability)
        - low_liquidity: Conservative (favor Sharpe, moderate CVaR)
        - normal: Default balanced weights

        Args:
            market_regime: Current market regime

        Returns:
            Tuple of (wealth_weight, sharpe_weight, cvar_weight)
        """
        # Define regime-specific weights (must sum to 1.0)
        regime_weights = {
            'trending_bull': {
                'wealth': 0.6,   # Aggressive: capture upside
                'sharpe': 0.2,   # Less concern about volatility
                'cvar': 0.2      # Minimal tail risk concern
            },
            'trending_bear': {
                'wealth': 0.2,   # Defensive: preserve capital
                'sharpe': 0.3,   # Maintain stability
                'cvar': 0.5      # High tail risk protection
            },
            'high_volatility': {
                'wealth': 0.2,   # Conservative during crisis
                'sharpe': 0.3,   # Seek stability
                'cvar': 0.5      # Maximum risk protection
            },
            'low_volatility': {
                'wealth': 0.3,   # Moderate growth
                'sharpe': 0.5,   # Prioritize risk-adjusted returns
                'cvar': 0.2      # Low tail risk in calm markets
            },
            'low_liquidity': {
                'wealth': 0.3,   # Cautious
                'sharpe': 0.4,   # Stability important
                'cvar': 0.3      # Moderate protection
            },
            'normal': {
                'wealth': 0.4,   # Balanced
                'sharpe': 0.3,   # Standard risk adjustment
                'cvar': 0.3      # Standard tail risk
            }
        }

        # Get weights for current regime (fallback to normal)
        weights = regime_weights.get(market_regime, regime_weights['normal'])

        return weights['wealth'], weights['sharpe'], weights['cvar']

    def calculate_reward(
        self,
        agent_returns: Dict[str, float],
        allocation: jnp.ndarray,
        macro_indicators: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate allocator reward with adaptive weights and reallocation cost

        Reward = wealth_weight * wealth_growth +
                sharpe_weight * portfolio_sharpe -
                cvar_weight * |portfolio_cvar| -
                reallocation_penalty

        Weights are dynamically adjusted based on market regime if enabled.

        Args:
            agent_returns: Returns from each agent
            allocation: Current allocation
            macro_indicators: Macro market indicators for regime detection

        Returns:
            Allocator reward
        """
        # Update adaptive weights based on market regime
        if self.enable_adaptive_weights and macro_indicators is not None:
            market_regime = self._detect_market_regime(macro_indicators)
            self.current_market_regime = market_regime
            self.regime_history.append(market_regime)

            # Get adaptive weights
            wealth_w, sharpe_w, cvar_w = self._get_adaptive_weights(market_regime)
            self.wealth_weight = wealth_w
            self.sharpe_weight = sharpe_w
            self.cvar_weight = cvar_w
        else:
            # Use default weights
            self.wealth_weight = self.default_wealth_weight
            self.sharpe_weight = self.default_sharpe_weight
            self.cvar_weight = self.default_cvar_weight

        # Blend with goal-specific risk targets if provided
        if self.goal_risk_targets:
            goal_risk_aversion = float(self.goal_risk_targets.get('risk_aversion', 1.0))
            goal_cvar = float(self.goal_risk_targets.get('cvar_budget', self.cvar_weight))

            # Normalize and clip to reasonable ranges
            goal_risk_aversion = jnp.clip(goal_risk_aversion, 0.1, 2.0)
            goal_cvar = jnp.clip(goal_cvar, 0.05, 1.0)

            # Blend current weights with goal preferences
            blend = jnp.clip(self.goal_alignment_strength, 0.0, 1.0)
            self.cvar_weight = float((1 - blend) * self.cvar_weight + blend * goal_risk_aversion)
            self.sharpe_weight = float((1 - blend) * self.sharpe_weight + blend * (1 - goal_cvar))
            self.wealth_weight = float((1 - blend) * self.wealth_weight + blend * (1 - goal_risk_aversion / 2))
        # Portfolio return (weighted average)
        returns_array = jnp.array([
            agent_returns.get('hft', 0.0),
            agent_returns.get('mft', 0.0),
            agent_returns.get('lft', 0.0)
        ])

        # Guard against NaN/inf in returns
        returns_array = jnp.nan_to_num(returns_array, nan=0.0, posinf=0.0, neginf=0.0)
        returns_array = jnp.clip(returns_array, -1.0, 1.0)

        portfolio_return = jnp.dot(allocation, returns_array)
        portfolio_return = jnp.nan_to_num(portfolio_return, nan=0.0, posinf=0.0, neginf=0.0)

        # Update wealth
        current_wealth = self.wealth_history[-1]
        new_wealth = current_wealth * (1 + portfolio_return)
        new_wealth = jnp.nan_to_num(new_wealth, nan=current_wealth, posinf=current_wealth, neginf=current_wealth)
        self.wealth_history.append(float(new_wealth))

        if len(self.wealth_history) > self.lookback_window:
            self.wealth_history.pop(0)

        # Wealth growth term
        wealth_growth = portfolio_return

        # Sharpe ratio term (if sufficient history)
        if len(self.wealth_history) >= 10:
            wealth_returns = jnp.diff(jnp.array(self.wealth_history)) / jnp.array(self.wealth_history[:-1])
            wealth_returns = jnp.nan_to_num(wealth_returns, nan=0.0, posinf=0.0, neginf=0.0)

            mean_return = jnp.mean(wealth_returns)
            std_return = jnp.std(wealth_returns)
            portfolio_sharpe = mean_return / (std_return + 1e-8)
            portfolio_sharpe = jnp.nan_to_num(portfolio_sharpe, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            portfolio_sharpe = jnp.array(0.0)

        # CVaR term (tail risk)
        if len(self.wealth_history) >= 20:
            wealth_returns = jnp.diff(jnp.array(self.wealth_history)) / jnp.array(self.wealth_history[:-1])
            wealth_returns = jnp.nan_to_num(wealth_returns, nan=0.0, posinf=0.0, neginf=0.0)

            var_threshold = jnp.quantile(wealth_returns, 0.05)
            var_threshold = jnp.nan_to_num(var_threshold, nan=0.0)

            tail_returns = wealth_returns[wealth_returns <= var_threshold]
            if len(tail_returns) > 0:
                portfolio_cvar = jnp.mean(tail_returns)
                portfolio_cvar = jnp.nan_to_num(portfolio_cvar, nan=0.0)
            else:
                portfolio_cvar = var_threshold

            portfolio_cvar = jnp.clip(portfolio_cvar, -10.0, 0.0)
        else:
            portfolio_cvar = jnp.array(0.0)

        # Reallocation cost penalty
        reallocation_penalty = 0.0
        if len(self.allocation_history) > 0:
            previous_allocation = jnp.array(self.allocation_history[-1])
            current_allocation = allocation

            # Calculate turnover (sum of absolute changes)
            turnover = jnp.sum(jnp.abs(current_allocation - previous_allocation))
            reallocation_penalty = self.reallocation_cost * turnover

            # Guard against NaN/inf
            reallocation_penalty = jnp.nan_to_num(reallocation_penalty, nan=0.0, posinf=0.0, neginf=0.0)
            reallocation_penalty = float(reallocation_penalty)

        # Combined reward with adaptive weights
        reward = (
            self.wealth_weight * wealth_growth +
            self.sharpe_weight * portfolio_sharpe -
            self.cvar_weight * jnp.abs(portfolio_cvar) -
            reallocation_penalty
        )

        return float(reward)

    def get_allocation_statistics(self) -> Dict[str, any]:
        """
        Get statistics about allocations and performance

        Returns:
            Dictionary of allocation statistics
        """
        if len(self.allocation_history) == 0:
            return {
                'mean_allocation': [1/3, 1/3, 1/3],
                'std_allocation': [0.0, 0.0, 0.0],
                'total_wealth': 1.0,
                'wealth_return': 0.0,
                'num_periods': 0,
                'current_regime': self.current_market_regime,
                'regime_distribution': {},
                'adaptive_weights': {
                    'wealth': self.wealth_weight,
                    'sharpe': self.sharpe_weight,
                    'cvar': self.cvar_weight
                },
                'guardrail_triggers': self.guardrail_trigger_count,
            }

        allocations = jnp.array(self.allocation_history)

        mean_allocation = jnp.mean(allocations, axis=0)
        std_allocation = jnp.std(allocations, axis=0)

        total_wealth = self.wealth_history[-1]
        initial_wealth = self.wealth_history[0]
        wealth_return = (total_wealth - initial_wealth) / initial_wealth

        # Calculate regime statistics if available
        regime_stats = {}
        if len(self.regime_history) > 0:
            from collections import Counter
            regime_counts = Counter(self.regime_history)
            total_periods = len(self.regime_history)
            regime_stats = {
                regime: count / total_periods
                for regime, count in regime_counts.items()
            }

        return {
            'mean_allocation': mean_allocation.tolist(),
            'std_allocation': std_allocation.tolist(),
            'total_wealth': float(total_wealth),
            'wealth_return': float(wealth_return),
            'num_periods': len(self.allocation_history),
            'current_regime': self.current_market_regime,
            'regime_distribution': regime_stats,
            'adaptive_weights': {
                'wealth': float(self.wealth_weight),
                'sharpe': float(self.sharpe_weight),
                'cvar': float(self.cvar_weight)
            },
            'guardrail_triggers': int(self.guardrail_trigger_count),
        }

    def get_regime_info(self) -> Dict[str, any]:
        """
        Get detailed information about market regime and adaptive weights

        Returns:
            Dictionary with regime and weight information
        """
        return {
            'current_regime': self.current_market_regime,
            'adaptive_enabled': self.enable_adaptive_weights,
            'current_weights': {
                'wealth': float(self.wealth_weight),
                'sharpe': float(self.sharpe_weight),
                'cvar': float(self.cvar_weight)
            },
            'default_weights': {
                'wealth': float(self.default_wealth_weight),
                'sharpe': float(self.default_sharpe_weight),
                'cvar': float(self.default_cvar_weight)
            },
            'reallocation_cost': float(self.reallocation_cost)
        }

    def reset(self):
        """Reset allocator state"""
        self.agent_performance_history = {
            'hft': [],
            'mft': [],
            'lft': []
        }
        self.allocation_history = []
        self.wealth_history = [1.0]
        self.current_market_regime = 'normal'
        self.regime_history = []

        # Reset weights to defaults
        self.wealth_weight = self.default_wealth_weight
        self.sharpe_weight = self.default_sharpe_weight
        self.cvar_weight = self.default_cvar_weight

        self.goal_allocation_targets = None
        self.goal_risk_targets = {}

    def set_goal_directive(
        self,
        allocation_targets: Optional[Dict[str, float]] = None,
        risk_targets: Optional[Dict[str, float]] = None,
        alignment_strength: float = 0.3
    ):
        """
        Update allocator with goal-driven targets coming from the planner.
        """
        if allocation_targets:
            self.goal_allocation_targets = dict(allocation_targets)
        if risk_targets:
            self.goal_risk_targets = dict(risk_targets)
            if 'allocation_floor' in risk_targets:
                self.min_allocation_floor = max(self.min_allocation_floor, float(risk_targets['allocation_floor']))
        self.goal_alignment_strength = float(alignment_strength)


class AllocationNetwork(nn.Module):
    """
    Neural network for learning allocation policy

    To be integrated with PPO/SAC for meta-level training
    """

    hidden_dims: List[int]
    num_agents: int

    @nn.compact
    def __call__(self, x):
        # Hidden layers
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)

        # Output layer (allocation logits)
        logits = nn.Dense(self.num_agents)(x)

        # Softmax to get allocation probabilities
        allocation = nn.softmax(logits)

        return allocation

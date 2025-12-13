"""
Order Book Environment for High-Frequency Trading
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional
import chex
from .base_env import BaseMarketEnv


class OrderBookEnv(BaseMarketEnv):
    """
    Order Book Environment for tick-level HFT simulation.

    Features:
    - Full order book depth (bids/asks)
    - Tick-by-tick execution
    - Market impact modeling
    - Queue position tracking
    """

    def __init__(
        self,
        num_assets: int = 1,
        depth_levels: int = 10,
        tick_size: float = 0.01,
        lot_size: int = 100,
        max_steps: int = 10000,
        **kwargs
    ):
        """
        Initialize order book environment.

        Args:
            num_assets: Number of assets (typically 1 for HFT)
            depth_levels: Number of price levels in order book
            tick_size: Minimum price increment
            lot_size: Minimum order size
            max_steps: Maximum steps per episode
            **kwargs: Additional arguments for base class
        """
        super().__init__(num_assets, **kwargs)

        self.depth_levels = depth_levels
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_steps = max_steps

        # Order book state
        self.bid_prices = None
        self.bid_volumes = None
        self.ask_prices = None
        self.ask_volumes = None
        self.mid_price = None
        self.spread = None

        # Market data
        self.orderbook_data = None
        self.current_tick = 0

        # Execution state
        self.inventory = 0.0
        self.last_trade_price = None

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, Any]]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.current_tick = 0
        self.done = False

        # Reset portfolio
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.inventory = 0.0

        # Initialize order book
        key, subkey = jax.random.split(key)
        self._init_orderbook(subkey)

        observation = self.get_observation()
        info = self.get_info()

        return observation, info

    def _init_orderbook(self, key: chex.PRNGKey) -> None:
        """Initialize order book with random state."""
        # Set initial mid price
        self.mid_price = 100.0

        # Generate bid side (below mid price)
        bid_offsets = jnp.arange(1, self.depth_levels + 1) * self.tick_size
        self.bid_prices = self.mid_price - bid_offsets

        # Generate ask side (above mid price)
        ask_offsets = jnp.arange(1, self.depth_levels + 1) * self.tick_size
        self.ask_prices = self.mid_price + ask_offsets

        # Random volumes at each level
        key1, key2 = jax.random.split(key)
        self.bid_volumes = jax.random.uniform(
            key1, (self.depth_levels,), minval=100, maxval=1000
        ) * self.lot_size
        self.ask_volumes = jax.random.uniform(
            key2, (self.depth_levels,), minval=100, maxval=1000
        ) * self.lot_size

        # Compute spread
        self.spread = self.ask_prices[0] - self.bid_prices[0]

    def step(
        self,
        action: chex.Array
    ) -> Tuple[chex.Array, float, bool, Dict[str, Any]]:
        """
        Execute one step in the order book environment.

        Args:
            action: [order_type, quantity, price_offset]
                - order_type: 0=market buy, 1=market sell, 2=limit buy, 3=limit sell
                - quantity: Order size in lots
                - price_offset: Price offset from mid (for limit orders)

        Returns:
            observation: Next observation
            reward: Reward signal
            done: Episode termination flag
            info: Additional information
        """
        # Parse action
        order_type = jnp.argmax(action[:4])  # One-hot encoded
        quantity = action[4] * 100  # Scale to reasonable quantity
        price_offset = action[5] * self.tick_size * 10  # Offset in ticks

        # Execute order
        executed_price, executed_qty, execution_cost = self._execute_order(
            order_type, quantity, price_offset
        )

        # Update inventory
        prev_inventory = self.inventory
        if order_type in [0, 2]:  # Buy orders
            self.inventory += executed_qty
        elif order_type in [1, 3]:  # Sell orders
            self.inventory -= executed_qty

        # Update order book (simulate market dynamics)
        self._update_orderbook()

        # Compute reward
        reward = self._compute_reward(
            prev_inventory,
            self.inventory,
            executed_price,
            execution_cost
        )

        # Update step counter
        self.current_step += 1
        self.current_tick += 1

        # Check if episode is done
        self.done = (
            self.current_step >= self.max_steps or
            jnp.abs(self.inventory) > 10000  # Inventory limit
        )

        # Get next observation
        observation = self.get_observation()

        # Compute portfolio value
        self.portfolio_value = self.cash + self.inventory * self.mid_price

        info = self.get_info()
        info.update({
            'executed_price': float(executed_price),
            'executed_qty': float(executed_qty),
            'execution_cost': float(execution_cost),
            'inventory': float(self.inventory),
            'mid_price': float(self.mid_price),
            'spread': float(self.spread)
        })

        return observation, float(reward), self.done, info

    def _execute_order(
        self,
        order_type: int,
        quantity: float,
        price_offset: float
    ) -> Tuple[float, float, float]:
        """
        Execute order and return execution details.

        Returns:
            executed_price: Average execution price
            executed_qty: Executed quantity
            execution_cost: Total execution cost
        """
        if order_type == 0:  # Market buy
            return self._execute_market_buy(quantity)
        elif order_type == 1:  # Market sell
            return self._execute_market_sell(quantity)
        elif order_type == 2:  # Limit buy
            limit_price = self.mid_price - price_offset
            return self._execute_limit_buy(quantity, limit_price)
        elif order_type == 3:  # Limit sell
            limit_price = self.mid_price + price_offset
            return self._execute_limit_sell(quantity, limit_price)
        else:
            return 0.0, 0.0, 0.0

    def _execute_market_buy(
        self,
        quantity: float
    ) -> Tuple[float, float, float]:
        """Execute market buy order (take liquidity from ask side)."""
        remaining_qty = quantity
        total_cost = 0.0
        executed_qty = 0.0

        # Walk through ask levels
        for level in range(self.depth_levels):
            if remaining_qty <= 0:
                break

            available_qty = self.ask_volumes[level]
            fill_qty = jnp.minimum(remaining_qty, available_qty)

            total_cost += fill_qty * self.ask_prices[level]
            executed_qty += fill_qty
            remaining_qty -= fill_qty

        if executed_qty > 0:
            avg_price = total_cost / executed_qty
            execution_cost = self.compute_transaction_costs(total_cost)
        else:
            avg_price = 0.0
            execution_cost = 0.0

        return avg_price, executed_qty, execution_cost

    def _execute_market_sell(
        self,
        quantity: float
    ) -> Tuple[float, float, float]:
        """Execute market sell order (take liquidity from bid side)."""
        remaining_qty = quantity
        total_proceeds = 0.0
        executed_qty = 0.0

        # Walk through bid levels
        for level in range(self.depth_levels):
            if remaining_qty <= 0:
                break

            available_qty = self.bid_volumes[level]
            fill_qty = jnp.minimum(remaining_qty, available_qty)

            total_proceeds += fill_qty * self.bid_prices[level]
            executed_qty += fill_qty
            remaining_qty -= fill_qty

        if executed_qty > 0:
            avg_price = total_proceeds / executed_qty
            execution_cost = self.compute_transaction_costs(total_proceeds)
        else:
            avg_price = 0.0
            execution_cost = 0.0

        return avg_price, executed_qty, execution_cost

    def _execute_limit_buy(
        self,
        quantity: float,
        limit_price: float
    ) -> Tuple[float, float, float]:
        """Execute limit buy order (provide liquidity on bid side)."""
        # Simplified: assume partial fill based on probability
        fill_prob = jnp.exp(-(limit_price - self.bid_prices[0]) / self.tick_size)
        fill_prob = jnp.clip(fill_prob, 0.0, 1.0)

        executed_qty = quantity * fill_prob
        avg_price = limit_price
        total_cost = executed_qty * avg_price

        execution_cost = self.compute_transaction_costs(total_cost)
        # Liquidity rebate (negative cost)
        execution_cost -= total_cost * 0.0002

        return avg_price, executed_qty, execution_cost

    def _execute_limit_sell(
        self,
        quantity: float,
        limit_price: float
    ) -> Tuple[float, float, float]:
        """Execute limit sell order (provide liquidity on ask side)."""
        # Simplified: assume partial fill based on probability
        fill_prob = jnp.exp(-(self.ask_prices[0] - limit_price) / self.tick_size)
        fill_prob = jnp.clip(fill_prob, 0.0, 1.0)

        executed_qty = quantity * fill_prob
        avg_price = limit_price
        total_proceeds = executed_qty * avg_price

        execution_cost = self.compute_transaction_costs(total_proceeds)
        # Liquidity rebate
        execution_cost -= total_proceeds * 0.0002

        return avg_price, executed_qty, execution_cost

    def _update_orderbook(self) -> None:
        """Update order book state (simulate market dynamics)."""
        # Simple random walk for mid price
        price_change = jax.random.normal(jax.random.PRNGKey(self.current_tick)) * 0.01
        self.mid_price += price_change

        # Update bid/ask prices
        self.bid_prices += price_change
        self.ask_prices += price_change

        # Random volume changes
        key = jax.random.PRNGKey(self.current_tick + 1)
        volume_change = jax.random.normal(key, (self.depth_levels,)) * 50
        self.bid_volumes = jnp.maximum(50, self.bid_volumes + volume_change)

        key = jax.random.PRNGKey(self.current_tick + 2)
        volume_change = jax.random.normal(key, (self.depth_levels,)) * 50
        self.ask_volumes = jnp.maximum(50, self.ask_volumes + volume_change)

        # Update spread
        self.spread = self.ask_prices[0] - self.bid_prices[0]

    def _compute_reward(
        self,
        prev_inventory: float,
        current_inventory: float,
        executed_price: float,
        execution_cost: float
    ) -> float:
        """
        Compute reward for HFT agent.

        Reward components:
        1. PnL from inventory changes
        2. Inventory risk penalty
        3. Execution cost penalty
        """
        # PnL from price changes
        inventory_pnl = current_inventory * (self.mid_price - executed_price)

        # Inventory risk penalty
        inventory_penalty = -0.01 * (current_inventory ** 2)

        # Execution cost penalty
        cost_penalty = -execution_cost

        # Total reward
        reward = inventory_pnl + inventory_penalty + cost_penalty

        return reward

    def get_observation(self) -> chex.Array:
        """
        Get current order book observation.

        Returns:
            observation: [bid_prices, bid_volumes, ask_prices, ask_volumes,
                         mid_price, spread, inventory, cash]
        """
        observation = jnp.concatenate([
            self.bid_prices / 100.0,  # Normalize prices
            self.bid_volumes / 1000.0,  # Normalize volumes
            self.ask_prices / 100.0,
            self.ask_volumes / 1000.0,
            jnp.array([self.mid_price / 100.0]),
            jnp.array([self.spread]),
            jnp.array([self.inventory / 1000.0]),
            jnp.array([self.cash / self.initial_capital])
        ])

        return observation

    def _action_to_trades(
        self,
        action: chex.Array,
        prices: chex.Array
    ) -> chex.Array:
        """Not used in order book env."""
        return jnp.zeros(1)

    def get_info(self) -> Dict[str, Any]:
        """
        Get environment information including orderbook-specific fields.

        Returns:
            info: Dictionary of environment information
        """
        info = super().get_info()
        info.update({
            'mid_price': float(self.mid_price) if self.mid_price is not None else 0.0,
            'spread': float(self.spread) if self.spread is not None else 0.0,
            'inventory': float(self.inventory)
        })
        return info

    def get_order_book_imbalance(self) -> float:
        """
        Compute order book imbalance indicator.

        Returns:
            imbalance: (bid_volume - ask_volume) / (bid_volume + ask_volume)
        """
        total_bid_volume = jnp.sum(self.bid_volumes)
        total_ask_volume = jnp.sum(self.ask_volumes)

        imbalance = (total_bid_volume - total_ask_volume) / (
            total_bid_volume + total_ask_volume + 1e-8
        )

        return imbalance

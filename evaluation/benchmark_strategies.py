"""
Benchmark Trading Strategies

Industry-standard strategies for comparison with our RL agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark strategy results"""
    name: str
    returns: np.ndarray
    capital_series: np.ndarray
    signals: np.ndarray
    description: str


class BenchmarkStrategies:
    """
    Collection of industry-standard trading strategies for benchmarking.

    包含的策略:
    1. Buy & Hold - 基准线
    2. Simple Moving Average Crossover - 简单趋势跟踪
    3. Momentum - 动量策略
    4. Mean Reversion - 均值回归
    5. Dual Momentum - 双动量（相对+绝对）
    6. 60/40 Portfolio - 传统配置
    7. Risk Parity - 风险平价
    8. Trend Following - 趋势跟踪
    """

    def __init__(self, initial_capital: float = 100000):
        """
        Initialize benchmark strategies.

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital

    def buy_and_hold(self, prices: np.ndarray) -> BenchmarkResult:
        """
        Buy and Hold strategy - the most basic benchmark.

        Args:
            prices: Asset prices

        Returns:
            BenchmarkResult with strategy performance
        """
        returns = np.diff(prices) / prices[:-1]

        # Full investment (100% long)
        signals = np.ones(len(prices))
        capital = self.initial_capital * prices / prices[0]

        return BenchmarkResult(
            name="Buy & Hold",
            returns=returns,
            capital_series=capital,
            signals=signals,
            description="Always fully invested (100% long position)"
        )

    def moving_average_crossover(
        self,
        prices: np.ndarray,
        fast_window: int = 20,
        slow_window: int = 50
    ) -> BenchmarkResult:
        """
        Simple Moving Average Crossover strategy.

        Buy when fast MA crosses above slow MA.
        Sell when fast MA crosses below slow MA.

        Args:
            prices: Asset prices
            fast_window: Fast MA window
            slow_window: Slow MA window

        Returns:
            BenchmarkResult
        """
        # Calculate moving averages
        fast_ma = self._moving_average(prices, fast_window)
        slow_ma = self._moving_average(prices, slow_window)

        # Generate signals: 1 = long, 0 = out
        signals = np.zeros(len(prices))
        signals[fast_ma > slow_ma] = 1.0

        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = price_returns * signals[:-1]

        # Calculate capital
        capital = self.initial_capital * np.cumprod(1 + strategy_returns)
        capital = np.insert(capital, 0, self.initial_capital)

        return BenchmarkResult(
            name=f"MA Cross ({fast_window}/{slow_window})",
            returns=strategy_returns,
            capital_series=capital,
            signals=signals,
            description=f"MA crossover with {fast_window}/{slow_window} periods"
        )

    def momentum_strategy(
        self,
        prices: np.ndarray,
        lookback: int = 20,
        holding_period: int = 5
    ) -> BenchmarkResult:
        """
        Momentum strategy - buy winners, sell losers.

        Args:
            prices: Asset prices
            lookback: Lookback period for momentum calculation
            holding_period: Holding period

        Returns:
            BenchmarkResult
        """
        # Calculate momentum
        momentum = np.zeros(len(prices))
        for i in range(lookback, len(prices)):
            momentum[i] = (prices[i] - prices[i - lookback]) / prices[i - lookback]

        # Generate signals: 1 if positive momentum, -1 if negative
        signals = np.sign(momentum)

        # Apply holding period (don't trade too frequently)
        for i in range(lookback + holding_period, len(signals), holding_period):
            signals[i:i+holding_period] = signals[i]

        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = price_returns * signals[:-1]

        # Calculate capital
        capital = self.initial_capital * np.cumprod(1 + strategy_returns)
        capital = np.insert(capital, 0, self.initial_capital)

        return BenchmarkResult(
            name=f"Momentum ({lookback}d)",
            returns=strategy_returns,
            capital_series=capital,
            signals=signals,
            description=f"Momentum strategy with {lookback}d lookback"
        )

    def mean_reversion(
        self,
        prices: np.ndarray,
        window: int = 20,
        std_threshold: float = 2.0
    ) -> BenchmarkResult:
        """
        Mean Reversion strategy using Bollinger Bands.

        Buy when price is below lower band (oversold).
        Sell when price is above upper band (overbought).

        Args:
            prices: Asset prices
            window: Window for moving average and std dev
            std_threshold: Number of standard deviations for bands

        Returns:
            BenchmarkResult
        """
        # Calculate moving average and bands
        ma = self._moving_average(prices, window)
        std = self._moving_std(prices, window)

        upper_band = ma + std_threshold * std
        lower_band = ma - std_threshold * std

        # Generate signals
        signals = np.zeros(len(prices))
        signals[prices < lower_band] = 1.0   # Buy when oversold
        signals[prices > upper_band] = -1.0  # Sell when overbought

        # Fill forward signals (hold position until next signal)
        for i in range(1, len(signals)):
            if signals[i] == 0:
                signals[i] = signals[i-1]

        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = price_returns * signals[:-1]

        # Calculate capital
        capital = self.initial_capital * np.cumprod(1 + strategy_returns)
        capital = np.insert(capital, 0, self.initial_capital)

        return BenchmarkResult(
            name=f"Mean Reversion (BB {std_threshold}σ)",
            returns=strategy_returns,
            capital_series=capital,
            signals=signals,
            description=f"Bollinger Bands mean reversion ({window}d, {std_threshold}σ)"
        )

    def dual_momentum(
        self,
        prices: np.ndarray,
        benchmark_prices: Optional[np.ndarray] = None,
        lookback: int = 63  # ~3 months
    ) -> BenchmarkResult:
        """
        Dual Momentum (Relative + Absolute).

        Combines:
        1. Relative momentum: outperform benchmark
        2. Absolute momentum: positive return

        Args:
            prices: Asset prices
            benchmark_prices: Benchmark prices (default: same as asset)
            lookback: Lookback period

        Returns:
            BenchmarkResult
        """
        if benchmark_prices is None:
            benchmark_prices = prices

        # Calculate returns
        asset_returns = (prices[lookback:] - prices[:-lookback]) / prices[:-lookback]
        bench_returns = (benchmark_prices[lookback:] - benchmark_prices[:-lookback]) / benchmark_prices[:-lookback]

        # Generate signals
        signals = np.zeros(len(prices))

        # Long if: (1) positive absolute return AND (2) outperforms benchmark
        for i in range(lookback, len(prices)):
            idx = i - lookback
            if asset_returns[idx] > 0 and asset_returns[idx] > bench_returns[idx]:
                signals[i] = 1.0

        # Calculate returns
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = price_returns * signals[:-1]

        # Calculate capital
        capital = self.initial_capital * np.cumprod(1 + strategy_returns)
        capital = np.insert(capital, 0, self.initial_capital)

        return BenchmarkResult(
            name="Dual Momentum",
            returns=strategy_returns,
            capital_series=capital,
            signals=signals,
            description=f"Dual momentum with {lookback}d lookback"
        )

    def portfolio_60_40(
        self,
        stock_prices: np.ndarray,
        bond_proxy_prices: Optional[np.ndarray] = None
    ) -> BenchmarkResult:
        """
        Classic 60/40 portfolio (60% stocks, 40% bonds).

        Args:
            stock_prices: Stock/equity prices
            bond_proxy_prices: Bond prices (if None, uses risk-free rate proxy)

        Returns:
            BenchmarkResult
        """
        stock_returns = np.diff(stock_prices) / stock_prices[:-1]

        if bond_proxy_prices is not None:
            bond_returns = np.diff(bond_proxy_prices) / bond_proxy_prices[:-1]
        else:
            # Proxy: assume 2% annual return for bonds
            bond_returns = np.ones(len(stock_returns)) * (0.02 / 252)

        # 60/40 allocation
        portfolio_returns = 0.6 * stock_returns + 0.4 * bond_returns

        # Calculate capital
        capital = self.initial_capital * np.cumprod(1 + portfolio_returns)
        capital = np.insert(capital, 0, self.initial_capital)

        signals = np.ones(len(stock_prices)) * 0.6  # 60% allocation

        return BenchmarkResult(
            name="60/40 Portfolio",
            returns=portfolio_returns,
            capital_series=capital,
            signals=signals,
            description="Classic 60% stocks / 40% bonds allocation"
        )

    def trend_following(
        self,
        prices: np.ndarray,
        fast_period: int = 10,
        slow_period: int = 50,
        atr_period: int = 14,
        atr_multiplier: float = 2.0
    ) -> BenchmarkResult:
        """
        Trend Following with ATR-based stops.

        Uses dual moving average with volatility-based position sizing.

        Args:
            prices: Asset prices
            fast_period: Fast trend period
            slow_period: Slow trend period
            atr_period: ATR calculation period
            atr_multiplier: Stop loss multiplier

        Returns:
            BenchmarkResult
        """
        # Calculate moving averages
        fast_ma = self._moving_average(prices, fast_period)
        slow_ma = self._moving_average(prices, slow_period)

        # Calculate ATR for position sizing
        high = prices * 1.01  # Simplified high
        low = prices * 0.99   # Simplified low
        atr = self._calculate_atr(high, low, prices, atr_period)

        # Generate signals with trend and volatility filter
        signals = np.zeros(len(prices))
        position_size = np.ones(len(prices))

        for i in range(slow_period, len(prices)):
            if fast_ma[i] > slow_ma[i]:
                signals[i] = 1.0
                # Size based on volatility (inverse volatility sizing)
                position_size[i] = min(1.0, 0.02 / (atr[i] / prices[i])) if atr[i] > 0 else 1.0
            elif fast_ma[i] < slow_ma[i]:
                signals[i] = -1.0
                position_size[i] = min(1.0, 0.02 / (atr[i] / prices[i])) if atr[i] > 0 else 1.0

        # Calculate returns with position sizing
        price_returns = np.diff(prices) / prices[:-1]
        strategy_returns = price_returns * signals[:-1] * position_size[:-1]

        # Calculate capital
        capital = self.initial_capital * np.cumprod(1 + strategy_returns)
        capital = np.insert(capital, 0, self.initial_capital)

        return BenchmarkResult(
            name="Trend Following",
            returns=strategy_returns,
            capital_series=capital,
            signals=signals,
            description=f"Trend following with ATR-based sizing ({fast_period}/{slow_period})"
        )

    # ==================== Helper Functions ====================

    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        ma = np.zeros(len(data))
        for i in range(window - 1, len(data)):
            ma[i] = np.mean(data[i - window + 1:i + 1])
        return ma

    def _moving_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving standard deviation"""
        std = np.zeros(len(data))
        for i in range(window - 1, len(data)):
            std[i] = np.std(data[i - window + 1:i + 1], ddof=1)
        return std

    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate Average True Range"""
        tr = np.zeros(len(close))

        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        # Calculate ATR as moving average of TR
        atr = self._moving_average(tr, period)
        return atr


def run_all_benchmarks(prices: np.ndarray, initial_capital: float = 100000) -> Dict[str, BenchmarkResult]:
    """
    Run all benchmark strategies and return results.

    Args:
        prices: Asset prices
        initial_capital: Starting capital

    Returns:
        Dictionary of strategy name -> BenchmarkResult
    """
    bench = BenchmarkStrategies(initial_capital=initial_capital)

    results = {
        'buy_hold': bench.buy_and_hold(prices),
        'ma_cross': bench.moving_average_crossover(prices),
        'momentum': bench.momentum_strategy(prices),
        'mean_reversion': bench.mean_reversion(prices),
        'dual_momentum': bench.dual_momentum(prices),
        'portfolio_60_40': bench.portfolio_60_40(prices),
        'trend_following': bench.trend_following(prices)
    }

    return results

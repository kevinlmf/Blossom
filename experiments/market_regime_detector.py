"""
Market Regime Detector

Identifies different market periods:
1. High Risk Period (Crisis/Crash) - 超大风险期
2. High Return Period (Bull Market) - 超大收益期
3. Stable Period (Normal Market) - 平稳期
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class MarketRegime(Enum):
    """Market regime types."""
    HIGH_RISK = "high_risk"  # Crisis/crash period
    HIGH_RETURN = "high_return"  # Bull market
    STABLE = "stable"  # Normal market
    UNKNOWN = "unknown"


@dataclass
class RegimePeriod:
    """Represents a period with specific market regime."""
    regime: MarketRegime
    start_date: datetime
    end_date: datetime
    start_idx: int
    end_idx: int
    volatility: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float


class MarketRegimeDetector:
    """
    Detects and classifies market regimes based on price data.

    Classification criteria:
    - High Risk: High volatility (>2x avg) + negative returns OR extreme drawdown (>20%)
    - High Return: High returns (>1.5x avg) + moderate volatility
    - Stable: Low volatility (<0.5x avg) + moderate returns
    """

    def __init__(
        self,
        volatility_window: int = 20,
        return_window: int = 20,
        high_vol_threshold: float = 2.0,  # Multiple of average volatility
        low_vol_threshold: float = 0.5,
        high_return_threshold: float = 1.5,  # Multiple of average return
        drawdown_threshold: float = 0.20  # 20% drawdown
    ):
        """
        Initialize the market regime detector.

        Args:
            volatility_window: Window for volatility calculation
            return_window: Window for return calculation
            high_vol_threshold: Threshold for high volatility (multiplier)
            low_vol_threshold: Threshold for low volatility (multiplier)
            high_return_threshold: Threshold for high returns (multiplier)
            drawdown_threshold: Threshold for extreme drawdown
        """
        self.volatility_window = volatility_window
        self.return_window = return_window
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.high_return_threshold = high_return_threshold
        self.drawdown_threshold = drawdown_threshold

    def detect_regimes(
        self,
        prices: np.ndarray,
        dates: Optional[List[datetime]] = None
    ) -> List[RegimePeriod]:
        """
        Detect market regimes from price data.

        Args:
            prices: Array of prices
            dates: Optional list of dates corresponding to prices

        Returns:
            List of RegimePeriod objects
        """
        if len(prices) < max(self.volatility_window, self.return_window) * 2:
            print(f"Warning: Not enough data for regime detection. Need at least {max(self.volatility_window, self.return_window) * 2} points.")
            return []

        # Compute returns
        returns = np.diff(prices) / prices[:-1]

        # Compute rolling volatility
        volatility = self._rolling_std(returns, self.volatility_window)

        # Compute rolling returns
        rolling_returns = self._rolling_mean(returns, self.return_window)

        # Compute drawdowns
        drawdowns = self._compute_drawdowns(prices)

        # Get average volatility and returns for normalization
        avg_vol = np.nanmean(volatility)
        avg_return = np.nanmean(np.abs(rolling_returns))

        # Classify each point
        regimes = []
        for i in range(len(volatility)):
            if np.isnan(volatility[i]) or np.isnan(rolling_returns[i]):
                regimes.append(MarketRegime.UNKNOWN)
                continue

            # Normalize metrics
            norm_vol = volatility[i] / avg_vol if avg_vol > 1e-8 else 1.0
            norm_return = rolling_returns[i] / avg_return if avg_return > 1e-8 else 0.0

            # Classify regime
            if norm_vol > self.high_vol_threshold and (rolling_returns[i] < 0 or drawdowns[i] < -self.drawdown_threshold):
                # High volatility + negative returns or extreme drawdown = Crisis
                regimes.append(MarketRegime.HIGH_RISK)
            elif rolling_returns[i] > 0 and norm_return > self.high_return_threshold and norm_vol < self.high_vol_threshold:
                # High positive returns + moderate volatility = Bull market
                regimes.append(MarketRegime.HIGH_RETURN)
            elif norm_vol < self.low_vol_threshold:
                # Low volatility = Stable
                regimes.append(MarketRegime.STABLE)
            else:
                # Default
                regimes.append(MarketRegime.UNKNOWN)

        # Group consecutive regimes into periods
        periods = self._group_regimes(
            regimes, volatility, rolling_returns, drawdowns, dates
        )

        return periods

    def detect_specific_periods(
        self,
        prices: np.ndarray,
        dates: Optional[List[datetime]] = None,
        min_period_length: int = 20
    ) -> Dict[str, List[RegimePeriod]]:
        """
        Detect and categorize specific market periods.

        Args:
            prices: Array of prices
            dates: Optional list of dates
            min_period_length: Minimum length for a valid period

        Returns:
            Dictionary mapping regime types to lists of periods
        """
        all_periods = self.detect_regimes(prices, dates)

        # Filter by minimum length and group by regime
        categorized = {
            'high_risk': [],
            'high_return': [],
            'stable': []
        }

        for period in all_periods:
            period_length = period.end_idx - period.start_idx

            if period_length < min_period_length:
                continue

            if period.regime == MarketRegime.HIGH_RISK:
                categorized['high_risk'].append(period)
            elif period.regime == MarketRegime.HIGH_RETURN:
                categorized['high_return'].append(period)
            elif period.regime == MarketRegime.STABLE:
                categorized['stable'].append(period)

        return categorized

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        result = np.full(len(data), np.nan)

        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1])

        return result

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean."""
        result = np.full(len(data), np.nan)

        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])

        return result

    def _compute_drawdowns(self, prices: np.ndarray) -> np.ndarray:
        """Compute drawdowns from peak."""
        cummax = np.maximum.accumulate(prices)
        drawdowns = (prices - cummax) / cummax

        return drawdowns

    def _group_regimes(
        self,
        regimes: List[MarketRegime],
        volatility: np.ndarray,
        returns: np.ndarray,
        drawdowns: np.ndarray,
        dates: Optional[List[datetime]] = None
    ) -> List[RegimePeriod]:
        """Group consecutive regimes into periods."""
        if not regimes:
            return []

        periods = []
        current_regime = regimes[0]
        start_idx = 0

        for i in range(1, len(regimes) + 1):
            # Check if regime changed or reached end
            if i == len(regimes) or regimes[i] != current_regime:
                # Skip UNKNOWN regimes
                if current_regime != MarketRegime.UNKNOWN:
                    # Compute period statistics
                    period_vol = np.nanmean(volatility[start_idx:i])
                    period_return = np.nanmean(returns[start_idx:i])
                    period_dd = np.nanmin(drawdowns[start_idx:i])

                    # Compute Sharpe ratio
                    period_returns = returns[start_idx:i]
                    period_sharpe = (
                        np.nanmean(period_returns) / np.nanstd(period_returns)
                        if np.nanstd(period_returns) > 1e-8 else 0.0
                    )

                    # Create period object
                    start_date = dates[start_idx] if dates and start_idx < len(dates) else None
                    end_date = dates[i - 1] if dates and i - 1 < len(dates) else None

                    period = RegimePeriod(
                        regime=current_regime,
                        start_date=start_date,
                        end_date=end_date,
                        start_idx=start_idx,
                        end_idx=i,
                        volatility=float(period_vol) if not np.isnan(period_vol) else 0.0,
                        avg_return=float(period_return) if not np.isnan(period_return) else 0.0,
                        sharpe_ratio=float(period_sharpe) if not np.isnan(period_sharpe) else 0.0,
                        max_drawdown=float(period_dd) if not np.isnan(period_dd) else 0.0
                    )

                    periods.append(period)

                # Move to next regime
                if i < len(regimes):
                    current_regime = regimes[i]
                    start_idx = i

        return periods

    def get_top_periods(
        self,
        periods: Dict[str, List[RegimePeriod]],
        n: int = 3
    ) -> Dict[str, List[RegimePeriod]]:
        """
        Get top N periods for each regime type based on characteristic strength.

        Args:
            periods: Dictionary of periods by regime
            n: Number of top periods to return

        Returns:
            Dictionary with top N periods for each regime
        """
        result = {}

        # High risk: sort by volatility + negative returns + drawdown
        if periods['high_risk']:
            high_risk_sorted = sorted(
                periods['high_risk'],
                key=lambda p: p.volatility - p.avg_return + abs(p.max_drawdown),
                reverse=True
            )
            result['high_risk'] = high_risk_sorted[:n]
        else:
            result['high_risk'] = []

        # High return: sort by returns
        if periods['high_return']:
            high_return_sorted = sorted(
                periods['high_return'],
                key=lambda p: p.avg_return,
                reverse=True
            )
            result['high_return'] = high_return_sorted[:n]
        else:
            result['high_return'] = []

        # Stable: sort by low volatility + positive Sharpe
        if periods['stable']:
            stable_sorted = sorted(
                periods['stable'],
                key=lambda p: -p.volatility + p.sharpe_ratio,
                reverse=True
            )
            result['stable'] = stable_sorted[:n]
        else:
            result['stable'] = []

        return result

    def print_period_summary(self, periods: Dict[str, List[RegimePeriod]]):
        """Print summary of detected periods."""
        print("\n" + "=" * 80)
        print("MARKET REGIME DETECTION SUMMARY")
        print("=" * 80)

        for regime_name, regime_periods in periods.items():
            print(f"\n{regime_name.upper().replace('_', ' ')} PERIODS: {len(regime_periods)} detected")
            print("-" * 80)

            for i, period in enumerate(regime_periods, 1):
                print(f"\nPeriod {i}:")
                if period.start_date and period.end_date:
                    print(f"  Date Range: {period.start_date.strftime('%Y-%m-%d')} to {period.end_date.strftime('%Y-%m-%d')}")
                print(f"  Index Range: [{period.start_idx}, {period.end_idx}]")
                print(f"  Length: {period.end_idx - period.start_idx} periods")
                print(f"  Volatility: {period.volatility:.4f}")
                print(f"  Avg Return: {period.avg_return:.4f}")
                print(f"  Sharpe Ratio: {period.sharpe_ratio:.4f}")
                print(f"  Max Drawdown: {period.max_drawdown:.2%}")

        print("\n" + "=" * 80)

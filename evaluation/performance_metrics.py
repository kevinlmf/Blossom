"""
Performance Metrics Computation

Comprehensive metrics for evaluating trading strategy performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import scipy.stats as stats


@dataclass
class StrategyMetrics:
    """Container for all strategy performance metrics"""

    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    avg_return: float

    # Risk metrics
    volatility: float
    downside_volatility: float
    max_drawdown: float
    avg_drawdown: float
    calmar_ratio: float

    # Risk-adjusted return metrics
    sharpe_ratio: float
    sortino_ratio: float
    omega_ratio: float

    # Risk measures
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float

    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Other metrics
    num_trades: int
    recovery_factor: float
    stability: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cumulative_return': self.cumulative_return,
            'avg_return': self.avg_return,
            'volatility': self.volatility,
            'downside_volatility': self.downside_volatility,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'omega_ratio': self.omega_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'var_99': self.var_99,
            'cvar_99': self.cvar_99,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'num_trades': self.num_trades,
            'recovery_factor': self.recovery_factor,
            'stability': self.stability
        }


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for trading strategies.
    """

    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            periods_per_year: Number of trading periods per year (default 252 for daily)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate_all_metrics(
        self,
        returns: np.ndarray,
        capital_series: Optional[np.ndarray] = None,
        trades: Optional[List[Dict]] = None
    ) -> StrategyMetrics:
        """
        Calculate all performance metrics.

        Args:
            returns: Array of period returns
            capital_series: Optional array of capital values over time
            trades: Optional list of individual trades

        Returns:
            StrategyMetrics object with all computed metrics
        """
        if len(returns) == 0:
            return self._get_empty_metrics()

        # Calculate return metrics
        total_return = self._calculate_total_return(returns)
        annualized_return = self._calculate_annualized_return(returns)
        cumulative_return = self._calculate_cumulative_return(returns)
        avg_return = float(np.mean(returns))

        # Calculate risk metrics
        volatility = self._calculate_volatility(returns)
        downside_volatility = self._calculate_downside_volatility(returns)

        if capital_series is not None:
            max_drawdown = self._calculate_max_drawdown(capital_series)
            avg_drawdown = self._calculate_avg_drawdown(capital_series)
        else:
            max_drawdown = self._calculate_max_drawdown_from_returns(returns)
            avg_drawdown = 0.0

        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)

        # Calculate risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        omega_ratio = self._calculate_omega_ratio(returns)

        # Calculate risk measures
        var_95, cvar_95 = self._calculate_var_cvar(returns, 0.95)
        var_99, cvar_99 = self._calculate_var_cvar(returns, 0.99)

        # Calculate trading metrics
        if trades is not None:
            win_rate, profit_factor, avg_win, avg_loss, max_consec_wins, max_consec_losses = \
                self._calculate_trading_metrics(trades)
            num_trades = len(trades)
        else:
            win_rate = self._estimate_win_rate_from_returns(returns)
            profit_factor = self._estimate_profit_factor_from_returns(returns)
            avg_win = float(np.mean(returns[returns > 0])) if len(returns[returns > 0]) > 0 else 0.0
            avg_loss = float(np.mean(returns[returns < 0])) if len(returns[returns < 0]) > 0 else 0.0
            max_consec_wins, max_consec_losses = self._calculate_consecutive_from_returns(returns)
            num_trades = len(returns)

        # Calculate other metrics
        recovery_factor = self._calculate_recovery_factor(total_return, max_drawdown)
        stability = self._calculate_stability(returns)

        return StrategyMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            avg_return=avg_return,
            volatility=volatility,
            downside_volatility=downside_volatility,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            calmar_ratio=calmar_ratio,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            omega_ratio=omega_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            num_trades=num_trades,
            recovery_factor=recovery_factor,
            stability=stability
        )

    # ==================== Return Metrics ====================

    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return percentage"""
        cumulative = np.prod(1 + returns) - 1
        return float(cumulative * 100)

    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return"""
        total_return = np.prod(1 + returns)
        n_periods = len(returns)
        n_years = n_periods / self.periods_per_year

        if n_years == 0:
            return 0.0

        annualized = (total_return ** (1 / n_years)) - 1
        return float(annualized * 100)

    def _calculate_cumulative_return(self, returns: np.ndarray) -> float:
        """Calculate cumulative return"""
        return float((np.prod(1 + returns) - 1) * 100)

    # ==================== Risk Metrics ====================

    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return 0.0
        vol = np.std(returns, ddof=1) * np.sqrt(self.periods_per_year)
        return float(vol * 100)

    def _calculate_downside_volatility(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate downside volatility (semi-deviation)"""
        downside_returns = returns[returns < threshold]
        if len(downside_returns) < 2:
            return 0.0
        downside_vol = np.std(downside_returns, ddof=1) * np.sqrt(self.periods_per_year)
        return float(downside_vol * 100)

    def _calculate_max_drawdown(self, capital_series: np.ndarray) -> float:
        """Calculate maximum drawdown from capital series"""
        if len(capital_series) < 2:
            return 0.0

        cummax = np.maximum.accumulate(capital_series)
        drawdown = (capital_series - cummax) / cummax
        max_dd = np.min(drawdown)
        return float(max_dd * 100)

    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """Calculate max drawdown from returns"""
        capital = np.cumprod(1 + returns)
        return self._calculate_max_drawdown(capital)

    def _calculate_avg_drawdown(self, capital_series: np.ndarray) -> float:
        """Calculate average drawdown"""
        if len(capital_series) < 2:
            return 0.0

        cummax = np.maximum.accumulate(capital_series)
        drawdowns = (capital_series - cummax) / cummax
        drawdowns = drawdowns[drawdowns < 0]

        if len(drawdowns) == 0:
            return 0.0

        return float(np.mean(drawdowns) * 100)

    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if abs(max_drawdown) < 1e-6:
            return 0.0
        return float(annualized_return / abs(max_drawdown))

    # ==================== Risk-Adjusted Metrics ====================

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess < 1e-8:
            return 0.0

        sharpe = (mean_excess / std_excess) * np.sqrt(self.periods_per_year)
        return float(sharpe)

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        mean_excess = np.mean(excess_returns)

        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            return 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std < 1e-8:
            return 0.0

        sortino = (mean_excess / downside_std) * np.sqrt(self.periods_per_year)
        return float(sortino)

    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if len(losses) == 0 or np.sum(losses) < 1e-8:
            return float('inf') if len(gains) > 0 else 0.0

        omega = np.sum(gains) / np.sum(losses)
        return float(omega)

    # ==================== Risk Measures ====================

    def _calculate_var_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk.

        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (VaR, CVaR)
        """
        if len(returns) == 0:
            return 0.0, 0.0

        # VaR: percentile of the loss distribution
        var = np.percentile(returns, (1 - confidence) * 100)

        # CVaR: average of returns below VaR
        losses_below_var = returns[returns <= var]
        cvar = np.mean(losses_below_var) if len(losses_below_var) > 0 else var

        return float(var * 100), float(cvar * 100)

    # ==================== Trading Metrics ====================

    def _calculate_trading_metrics(self, trades: List[Dict]) -> Tuple[float, float, float, float, int, int]:
        """Calculate trading-specific metrics from trade list"""
        if not trades:
            return 0.0, 0.0, 0.0, 0.0, 0, 0

        profits = [t['pnl'] for t in trades if 'pnl' in t]
        if not profits:
            return 0.0, 0.0, 0.0, 0.0, 0, 0

        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        # Win rate
        win_rate = len(wins) / len(profits) if profits else 0.0

        # Profit factor
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Average win/loss
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        # Consecutive wins/losses
        max_consec_wins, max_consec_losses = self._calculate_consecutive_trades(profits)

        return win_rate, profit_factor, float(avg_win), float(avg_loss), max_consec_wins, max_consec_losses

    def _estimate_win_rate_from_returns(self, returns: np.ndarray) -> float:
        """Estimate win rate from returns"""
        if len(returns) == 0:
            return 0.0
        positive = np.sum(returns > 0)
        return float(positive / len(returns))

    def _estimate_profit_factor_from_returns(self, returns: np.ndarray) -> float:
        """Estimate profit factor from returns"""
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))

        if losses < 1e-8:
            return float('inf') if gains > 0 else 0.0

        return float(gains / losses)

    def _calculate_consecutive_from_returns(self, returns: np.ndarray) -> Tuple[int, int]:
        """Calculate max consecutive wins/losses from returns"""
        if len(returns) == 0:
            return 0, 0

        wins_losses = (returns > 0).astype(int)
        wins_losses[returns < 0] = -1

        return self._calculate_consecutive_trades(wins_losses)

    def _calculate_consecutive_trades(self, results: np.ndarray) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses"""
        if len(results) == 0:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for result in results:
            if result > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif result < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    # ==================== Other Metrics ====================

    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        if abs(max_drawdown) < 1e-6:
            return 0.0
        return float(total_return / abs(max_drawdown))

    def _calculate_stability(self, returns: np.ndarray) -> float:
        """
        Calculate stability coefficient (R² of linear regression on cumulative returns).
        Higher values indicate more stable growth.
        """
        if len(returns) < 2:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        x = np.arange(len(cumulative))

        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(x, cumulative)

        return float(r_value ** 2)

    def _get_empty_metrics(self) -> StrategyMetrics:
        """Return empty metrics when no data available"""
        return StrategyMetrics(
            total_return=0.0,
            annualized_return=0.0,
            cumulative_return=0.0,
            avg_return=0.0,
            volatility=0.0,
            downside_volatility=0.0,
            max_drawdown=0.0,
            avg_drawdown=0.0,
            calmar_ratio=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            omega_ratio=0.0,
            var_95=0.0,
            cvar_95=0.0,
            var_99=0.0,
            cvar_99=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            num_trades=0,
            recovery_factor=0.0,
            stability=0.0
        )

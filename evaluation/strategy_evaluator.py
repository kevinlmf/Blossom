"""
Strategy Evaluator

Main class for comprehensive strategy evaluation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .performance_metrics import PerformanceMetrics, StrategyMetrics
from .visualization import EvaluationVisualizer


class StrategyEvaluator:
    """
    Comprehensive strategy evaluator that combines metrics calculation,
    visualization, and reporting.
    """

    def __init__(
        self,
        output_dir: str = "outputs/evaluation",
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize strategy evaluator.

        Args:
            output_dir: Directory to save evaluation outputs
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_calculator = PerformanceMetrics(
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year
        )
        self.visualizer = EvaluationVisualizer(output_dir=str(self.output_dir))

        print(f"Strategy Evaluator initialized. Output: {self.output_dir}")

    def evaluate_strategy(
        self,
        returns: np.ndarray,
        capital_series: Optional[np.ndarray] = None,
        trades: Optional[List[Dict]] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        regime: Optional[str] = None,
        agent_name: Optional[str] = None,
        save_report: bool = True
    ) -> StrategyMetrics:
        """
        Evaluate a single strategy comprehensively.

        Args:
            returns: Array of period returns
            capital_series: Optional array of capital values
            trades: Optional list of individual trades
            benchmark_returns: Optional benchmark returns for comparison
            regime: Market regime name
            agent_name: Agent name (hft, mft, lft)
            save_report: Whether to save evaluation report

        Returns:
            StrategyMetrics object with all metrics
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING STRATEGY")
        if regime:
            print(f"Regime: {regime.upper()}")
        if agent_name:
            print(f"Agent: {agent_name.upper()}")
        print(f"{'='*80}")

        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            returns=returns,
            capital_series=capital_series,
            trades=trades
        )

        # Print metrics summary
        self._print_metrics_summary(metrics)

        # Compare with benchmark if provided
        if benchmark_returns is not None:
            self._print_benchmark_comparison(returns, benchmark_returns, metrics)

        if save_report:
            # Generate visualization
            if capital_series is None:
                capital_series = np.cumprod(1 + returns) * 100000  # Assume $100k start

            benchmark_capital = None
            if benchmark_returns is not None:
                benchmark_capital = np.cumprod(1 + benchmark_returns) * 100000

            # Create comprehensive report
            report_name = self._get_report_filename(regime, agent_name)
            self.visualizer.plot_comprehensive_report(
                returns=returns,
                capital_series=capital_series,
                metrics=metrics.to_dict(),
                benchmark_returns=benchmark_returns,
                benchmark_capital=benchmark_capital,
                regime=regime,
                save_name=report_name
            )

            # Save metrics to JSON
            self._save_metrics_json(metrics, regime, agent_name)

        return metrics

    def evaluate_multiple_agents(
        self,
        agent_results: Dict[str, Dict],
        regime: Optional[str] = None,
        save_report: bool = True
    ) -> Dict[str, StrategyMetrics]:
        """
        Evaluate multiple agents and compare performance.

        Args:
            agent_results: Dictionary mapping agent names to their results
                          Each result should have 'returns' and optionally 'capital', 'trades'
            regime: Market regime name
            save_report: Whether to save comparison report

        Returns:
            Dictionary mapping agent names to StrategyMetrics
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING MULTIPLE AGENTS")
        if regime:
            print(f"Regime: {regime.upper()}")
        print(f"Agents: {', '.join([a.upper() for a in agent_results.keys()])}")
        print(f"{'='*80}")

        all_metrics = {}

        # Evaluate each agent
        for agent_name, results in agent_results.items():
            returns = results['returns']
            capital = results.get('capital', None)
            trades = results.get('trades', None)

            metrics = self.metrics_calculator.calculate_all_metrics(
                returns=returns,
                capital_series=capital,
                trades=trades
            )
            all_metrics[agent_name] = metrics

            print(f"\n{agent_name.upper()} Performance:")
            print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
            print(f"  Total Return: {metrics.total_return:.2f}%")
            print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")

        if save_report:
            # Create agent comparison visualization
            agent_returns = {name: results['returns']
                           for name, results in agent_results.items()}

            report_name = f"agent_comparison_{regime}.png" if regime else "agent_comparison.png"
            self.visualizer.plot_agent_comparison(
                agent_returns=agent_returns,
                save_name=report_name
            )

        print(f"\n{'='*80}")

        return all_metrics

    def evaluate_multiple_regimes(
        self,
        regime_results: Dict[str, Dict],
        save_report: bool = True
    ) -> Dict[str, StrategyMetrics]:
        """
        Evaluate performance across multiple market regimes.

        Args:
            regime_results: Dictionary mapping regime names to their results
                          Each result should have 'returns' and optionally 'capital', 'trades'
            save_report: Whether to save comparison report

        Returns:
            Dictionary mapping regime names to StrategyMetrics
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING MULTIPLE REGIMES")
        print(f"Regimes: {', '.join([r.upper() for r in regime_results.keys()])}")
        print(f"{'='*80}")

        all_metrics = {}

        # Evaluate each regime
        for regime_name, results in regime_results.items():
            returns = results['returns']
            capital = results.get('capital', None)
            trades = results.get('trades', None)

            metrics = self.metrics_calculator.calculate_all_metrics(
                returns=returns,
                capital_series=capital,
                trades=trades
            )
            all_metrics[regime_name] = metrics

            print(f"\n{regime_name.upper()} Performance:")
            print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
            print(f"  Total Return: {metrics.total_return:.2f}%")
            print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")

        if save_report:
            # Create regime comparison visualization
            regime_metrics = {name: metrics.to_dict()
                            for name, metrics in all_metrics.items()}

            self.visualizer.plot_regime_comparison(
                regime_metrics=regime_metrics,
                save_name="regime_comparison.png"
            )

        print(f"\n{'='*80}")

        return all_metrics

    def _print_metrics_summary(self, metrics: StrategyMetrics):
        """Print formatted metrics summary"""
        print(f"\n📊 PERFORMANCE METRICS")
        print(f"{'─'*80}")

        print(f"\n🎯 Return Metrics:")
        print(f"  Total Return:       {metrics.total_return:>10.2f}%")
        print(f"  Annualized Return:  {metrics.annualized_return:>10.2f}%")
        print(f"  Average Return:     {metrics.avg_return * 100:>10.4f}%")

        print(f"\n⚠️  Risk Metrics:")
        print(f"  Volatility:         {metrics.volatility:>10.2f}%")
        print(f"  Downside Vol.:      {metrics.downside_volatility:>10.2f}%")
        print(f"  Max Drawdown:       {metrics.max_drawdown:>10.2f}%")

        print(f"\n📈 Risk-Adjusted Returns:")
        print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:>10.3f}")
        print(f"  Sortino Ratio:      {metrics.sortino_ratio:>10.3f}")
        print(f"  Calmar Ratio:       {metrics.calmar_ratio:>10.3f}")
        print(f"  Omega Ratio:        {metrics.omega_ratio:>10.3f}")

        print(f"\n🎲 Risk Measures:")
        print(f"  VaR (95%):          {metrics.var_95:>10.2f}%")
        print(f"  CVaR (95%):         {metrics.cvar_95:>10.2f}%")
        print(f"  VaR (99%):          {metrics.var_99:>10.2f}%")
        print(f"  CVaR (99%):         {metrics.cvar_99:>10.2f}%")

        print(f"\n💰 Trading Metrics:")
        print(f"  Win Rate:           {metrics.win_rate * 100:>10.2f}%")
        print(f"  Profit Factor:      {metrics.profit_factor:>10.3f}")
        print(f"  Avg Win:            {metrics.avg_win:>10.4f}")
        print(f"  Avg Loss:           {metrics.avg_loss:>10.4f}")
        print(f"  Max Consec. Wins:   {metrics.max_consecutive_wins:>10d}")
        print(f"  Max Consec. Losses: {metrics.max_consecutive_losses:>10d}")
        print(f"  Number of Trades:   {metrics.num_trades:>10d}")

        print(f"\n✨ Other Metrics:")
        print(f"  Recovery Factor:    {metrics.recovery_factor:>10.3f}")
        print(f"  Stability:          {metrics.stability:>10.3f}")

        print(f"\n{'─'*80}")

    def _print_benchmark_comparison(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        strategy_metrics: StrategyMetrics
    ):
        """Print comparison with benchmark"""
        benchmark_metrics = self.metrics_calculator.calculate_all_metrics(benchmark_returns)

        print(f"\n📊 BENCHMARK COMPARISON (Buy & Hold)")
        print(f"{'─'*80}")

        metrics_to_compare = [
            ('Total Return', 'total_return', '%'),
            ('Annualized Return', 'annualized_return', '%'),
            ('Volatility', 'volatility', '%'),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Max Drawdown', 'max_drawdown', '%'),
            ('Calmar Ratio', 'calmar_ratio', ''),
        ]

        print(f"\n{'Metric':<25} {'Strategy':>15} {'Benchmark':>15} {'Difference':>15}")
        print(f"{'─'*72}")

        for name, key, unit in metrics_to_compare:
            strategy_val = getattr(strategy_metrics, key)
            benchmark_val = getattr(benchmark_metrics, key)
            diff = strategy_val - benchmark_val

            # Format difference with +/- and color indicator
            diff_str = f"{diff:+.2f}{unit}"
            indicator = "✅" if diff > 0 else "❌" if diff < 0 else "➖"

            print(f"{name:<25} {strategy_val:>13.2f}{unit:>2} "
                  f"{benchmark_val:>13.2f}{unit:>2} "
                  f"{diff_str:>13} {indicator}")

        print(f"{'─'*80}")

    def _save_metrics_json(
        self,
        metrics: StrategyMetrics,
        regime: Optional[str],
        agent_name: Optional[str]
    ):
        """Save metrics to JSON file"""
        metrics_dict = metrics.to_dict()

        # Add metadata
        metrics_dict['timestamp'] = datetime.now().isoformat()
        if regime:
            metrics_dict['regime'] = regime
        if agent_name:
            metrics_dict['agent'] = agent_name

        # Create filename
        filename = self._get_json_filename(regime, agent_name)
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"\n💾 Metrics saved: {filepath}")

    def _get_report_filename(self, regime: Optional[str], agent_name: Optional[str]) -> str:
        """Generate report filename"""
        parts = ['strategy_evaluation']
        if regime:
            parts.append(regime)
        if agent_name:
            parts.append(agent_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
        return '_'.join(parts) + '.png'

    def _get_json_filename(self, regime: Optional[str], agent_name: Optional[str]) -> str:
        """Generate JSON filename"""
        parts = ['metrics']
        if regime:
            parts.append(regime)
        if agent_name:
            parts.append(agent_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
        return '_'.join(parts) + '.json'

    def generate_summary_report(
        self,
        all_results: Dict[str, Dict],
        report_name: str = "summary_report.txt"
    ):
        """
        Generate a comprehensive text summary report.

        Args:
            all_results: Dictionary with all evaluation results
            report_name: Name of the report file
        """
        filepath = self.output_dir / report_name

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-FREQUENCY TRADING SYSTEM - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall summary
            if 'overall' in all_results:
                f.write("OVERALL SYSTEM PERFORMANCE\n")
                f.write("-"*80 + "\n")
                overall = all_results['overall']
                for key, value in overall.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Regime-wise results
            if 'regimes' in all_results:
                f.write("PERFORMANCE BY MARKET REGIME\n")
                f.write("-"*80 + "\n")
                for regime, metrics in all_results['regimes'].items():
                    f.write(f"\n{regime.upper()}:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Agent-wise results
            if 'agents' in all_results:
                f.write("PERFORMANCE BY AGENT\n")
                f.write("-"*80 + "\n")
                for agent, metrics in all_results['agents'].items():
                    f.write(f"\n{agent.upper()}:\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"\n📄 Summary report saved: {filepath}")

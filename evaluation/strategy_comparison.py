"""
Comprehensive Strategy Comparison Framework

Compare our RL agent against industry benchmark strategies with:
1. Statistical significance testing
2. Multiple performance metrics
3. Risk-adjusted comparisons
4. Comprehensive visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .performance_metrics import PerformanceMetrics, StrategyMetrics
from .benchmark_strategies import BenchmarkStrategies, BenchmarkResult


@dataclass
class ComparisonResult:
    """Results from strategy comparison"""
    our_strategy: StrategyMetrics
    benchmark_strategies: Dict[str, StrategyMetrics]
    statistical_tests: Dict[str, Dict[str, float]]
    rankings: Dict[str, int]  # Rank by different metrics
    superiority_score: float  # How much better than average benchmark


class StrategyComparator:
    """
    Comprehensive comparison framework for validating our strategy
    against industry benchmarks.

    核心目标：
    1. 证明我们的策略显著优于传统策略
    2. 提供统计学证据
    3. 量化改进幅度
    """

    def __init__(self, output_dir: str = "outputs/comparison"):
        """
        Initialize strategy comparator.

        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_calculator = PerformanceMetrics()
        self.benchmark_strategies = BenchmarkStrategies()

        print(f"Strategy Comparator initialized.")
        print(f"Output: {self.output_dir}")

    def comprehensive_comparison(
        self,
        our_returns: np.ndarray,
        prices: np.ndarray,
        initial_capital: float = 100000,
        save_report: bool = True
    ) -> ComparisonResult:
        """
        Run comprehensive comparison against all benchmarks.

        Args:
            our_returns: Returns from our RL agent
            prices: Asset prices
            initial_capital: Starting capital
            save_report: Whether to save detailed report

        Returns:
            ComparisonResult
        """
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE STRATEGY COMPARISON")
        print(f"{'='*80}")

        # Calculate our strategy metrics
        print(f"\n📊 Evaluating our RL strategy...")
        our_metrics = self.metrics_calculator.calculate_all_metrics(our_returns)

        print(f"  Sharpe: {our_metrics.sharpe_ratio:.3f}")
        print(f"  Return: {our_metrics.total_return:.2f}%")
        print(f"  Max DD: {our_metrics.max_drawdown:.2f}%")

        # Run all benchmark strategies
        print(f"\n📊 Running {7} benchmark strategies...")
        benchmark_results = self._run_all_benchmarks(prices, initial_capital)

        # Calculate benchmark metrics
        benchmark_metrics = {}
        for name, result in benchmark_results.items():
            metrics = self.metrics_calculator.calculate_all_metrics(result.returns)
            benchmark_metrics[name] = metrics
            print(f"  {result.name:<25} Sharpe: {metrics.sharpe_ratio:>6.3f}  Return: {metrics.total_return:>7.2f}%")

        # Statistical significance tests
        print(f"\n🧪 Running statistical significance tests...")
        stat_tests = self._run_statistical_tests(our_returns, benchmark_results)

        # Calculate rankings
        print(f"\n🏆 Calculating rankings...")
        rankings = self._calculate_rankings(our_metrics, benchmark_metrics)

        # Calculate superiority score
        superiority = self._calculate_superiority_score(our_metrics, benchmark_metrics)

        # Print summary
        self._print_comparison_summary(our_metrics, benchmark_metrics, rankings, superiority)

        # Create comparison result
        result = ComparisonResult(
            our_strategy=our_metrics,
            benchmark_strategies=benchmark_metrics,
            statistical_tests=stat_tests,
            rankings=rankings,
            superiority_score=superiority
        )

        if save_report:
            # Save detailed report
            self._save_comparison_report(result)

            # Create visualizations
            self._create_comparison_visualizations(our_metrics, benchmark_metrics, our_returns, benchmark_results)

        return result

    def _run_all_benchmarks(
        self,
        prices: np.ndarray,
        initial_capital: float
    ) -> Dict[str, BenchmarkResult]:
        """Run all benchmark strategies"""
        bench = BenchmarkStrategies(initial_capital=initial_capital)

        return {
            'buy_hold': bench.buy_and_hold(prices),
            'ma_cross': bench.moving_average_crossover(prices, fast_window=20, slow_window=50),
            'momentum': bench.momentum_strategy(prices, lookback=20),
            'mean_reversion': bench.mean_reversion(prices, window=20),
            'dual_momentum': bench.dual_momentum(prices, lookback=63),
            'portfolio_60_40': bench.portfolio_60_40(prices),
            'trend_following': bench.trend_following(prices)
        }

    def _run_statistical_tests(
        self,
        our_returns: np.ndarray,
        benchmark_results: Dict[str, BenchmarkResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Run statistical significance tests.

        Tests:
        1. T-test: Are our returns significantly different?
        2. Mann-Whitney U: Non-parametric alternative
        3. Sharpe Ratio test: Is our Sharpe significantly better?

        Returns:
            Dict of test results for each benchmark
        """
        tests = {}

        for name, result in benchmark_results.items():
            bench_returns = result.returns

            # Ensure same length
            min_len = min(len(our_returns), len(bench_returns))
            our_r = our_returns[:min_len]
            bench_r = bench_returns[:min_len]

            # T-test
            t_stat, t_pval = stats.ttest_ind(our_r, bench_r)

            # Mann-Whitney U test (non-parametric)
            u_stat, u_pval = stats.mannwhitneyu(our_r, bench_r, alternative='greater')

            # Sharpe ratio test (using Jobson-Korkie test approximation)
            our_sharpe = np.mean(our_r) / np.std(our_r) if np.std(our_r) > 0 else 0
            bench_sharpe = np.mean(bench_r) / np.std(bench_r) if np.std(bench_r) > 0 else 0

            sharpe_diff = our_sharpe - bench_sharpe
            sharpe_se = self._calculate_sharpe_se(our_r, bench_r)
            sharpe_z = sharpe_diff / sharpe_se if sharpe_se > 0 else 0
            sharpe_pval = 1 - stats.norm.cdf(sharpe_z)

            tests[name] = {
                't_statistic': float(t_stat),
                't_pvalue': float(t_pval),
                'u_statistic': float(u_stat),
                'u_pvalue': float(u_pval),
                'sharpe_z_score': float(sharpe_z),
                'sharpe_pvalue': float(sharpe_pval),
                'significant_at_5pct': t_pval < 0.05 and u_pval < 0.05
            }

        return tests

    def _calculate_sharpe_se(self, returns1: np.ndarray, returns2: np.ndarray) -> float:
        """
        Calculate standard error of difference in Sharpe ratios.

        Uses Jobson-Korkie method.
        """
        n = len(returns1)
        if n < 2:
            return 0.0

        sr1 = np.mean(returns1) / np.std(returns1) if np.std(returns1) > 0 else 0
        sr2 = np.mean(returns2) / np.std(returns2) if np.std(returns2) > 0 else 0

        var_sr1 = (1 + 0.5 * sr1**2) / n
        var_sr2 = (1 + 0.5 * sr2**2) / n

        return np.sqrt(var_sr1 + var_sr2)

    def _calculate_rankings(
        self,
        our_metrics: StrategyMetrics,
        benchmark_metrics: Dict[str, StrategyMetrics]
    ) -> Dict[str, int]:
        """
        Calculate rankings across different metrics.

        Returns:
            Dict mapping metric name to our rank (1 = best)
        """
        # Metrics to rank (higher is better)
        metrics_higher_better = ['sharpe_ratio', 'total_return', 'sortino_ratio', 'calmar_ratio', 'win_rate']

        # Metrics to rank (lower is better - invert)
        metrics_lower_better = ['max_drawdown', 'volatility']

        rankings = {}

        for metric in metrics_higher_better:
            values = {
                'Our Strategy': getattr(our_metrics, metric),
                **{name: getattr(m, metric) for name, m in benchmark_metrics.items()}
            }
            sorted_strategies = sorted(values.items(), key=lambda x: x[1], reverse=True)
            our_rank = next(i for i, (name, _) in enumerate(sorted_strategies, 1) if name == 'Our Strategy')
            rankings[metric] = our_rank

        for metric in metrics_lower_better:
            values = {
                'Our Strategy': abs(getattr(our_metrics, metric)),  # Use absolute value
                **{name: abs(getattr(m, metric)) for name, m in benchmark_metrics.items()}
            }
            sorted_strategies = sorted(values.items(), key=lambda x: x[1])  # Lower is better
            our_rank = next(i for i, (name, _) in enumerate(sorted_strategies, 1) if name == 'Our Strategy')
            rankings[metric] = our_rank

        return rankings

    def _calculate_superiority_score(
        self,
        our_metrics: StrategyMetrics,
        benchmark_metrics: Dict[str, StrategyMetrics]
    ) -> float:
        """
        Calculate overall superiority score.

        Score = (Our Sharpe - Average Benchmark Sharpe) / Std(Benchmark Sharpes)

        Returns:
            Z-score indicating how many standard deviations better we are
        """
        benchmark_sharpes = [m.sharpe_ratio for m in benchmark_metrics.values()]

        avg_bench_sharpe = np.mean(benchmark_sharpes)
        std_bench_sharpe = np.std(benchmark_sharpes)

        if std_bench_sharpe == 0:
            return 0.0

        superiority = (our_metrics.sharpe_ratio - avg_bench_sharpe) / std_bench_sharpe

        return float(superiority)

    def _print_comparison_summary(
        self,
        our_metrics: StrategyMetrics,
        benchmark_metrics: Dict[str, StrategyMetrics],
        rankings: Dict[str, int],
        superiority: float
    ):
        """Print formatted comparison summary"""
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE COMPARISON SUMMARY")
        print(f"{'='*80}")

        # Rankings
        print(f"\n🏆 RANKINGS (out of {len(benchmark_metrics) + 1} strategies)")
        print(f"{'─'*80}")
        print(f"{'Metric':<30} {'Our Rank':>15} {'Status':>20}")
        print(f"{'─'*80}")

        for metric, rank in rankings.items():
            status = "🥇 BEST" if rank == 1 else f"#{rank}"
            metric_display = metric.replace('_', ' ').title()
            print(f"{metric_display:<30} {rank:>15} {status:>20}")

        # Superiority score
        print(f"\n💪 SUPERIORITY SCORE")
        print(f"{'─'*80}")
        print(f"Z-Score vs. Benchmarks: {superiority:+.2f}σ")

        if superiority > 2.0:
            interpretation = "🌟 EXCEPTIONAL - FAR SUPERIOR to benchmarks"
        elif superiority > 1.0:
            interpretation = "✅ EXCELLENT - Significantly better than benchmarks"
        elif superiority > 0.5:
            interpretation = "👍 GOOD - Better than average benchmark"
        elif superiority > 0:
            interpretation = "➖ MARGINAL - Slightly better than benchmarks"
        else:
            interpretation = "⚠️  UNDERPERFORMING - Below benchmark average"

        print(f"Interpretation: {interpretation}")

        # Best benchmark comparison
        best_benchmark_name = max(
            benchmark_metrics.items(),
            key=lambda x: x[1].sharpe_ratio
        )[0]
        best_benchmark = benchmark_metrics[best_benchmark_name]

        print(f"\n🆚 COMPARISON WITH BEST BENCHMARK ({best_benchmark_name})")
        print(f"{'─'*80}")
        print(f"{'Metric':<30} {'Our Strategy':>15} {'Best Benchmark':>15} {'Difference':>15}")
        print(f"{'─'*80}")

        comparisons = [
            ('Sharpe Ratio', our_metrics.sharpe_ratio, best_benchmark.sharpe_ratio),
            ('Total Return (%)', our_metrics.total_return, best_benchmark.total_return),
            ('Max Drawdown (%)', our_metrics.max_drawdown, best_benchmark.max_drawdown),
            ('Sortino Ratio', our_metrics.sortino_ratio, best_benchmark.sortino_ratio),
            ('Calmar Ratio', our_metrics.calmar_ratio, best_benchmark.calmar_ratio),
            ('Win Rate (%)', our_metrics.win_rate * 100, best_benchmark.win_rate * 100),
        ]

        for metric, our_val, bench_val in comparisons:
            diff = our_val - bench_val
            indicator = "✅" if diff > 0 else "❌"
            print(f"{metric:<30} {our_val:>15.2f} {bench_val:>15.2f} {diff:>14.2f} {indicator}")

        print(f"{'='*80}")

    def _create_comparison_visualizations(
        self,
        our_metrics: StrategyMetrics,
        benchmark_metrics: Dict[str, StrategyMetrics],
        our_returns: np.ndarray,
        benchmark_results: Dict[str, BenchmarkResult]
    ):
        """Create comprehensive comparison visualizations"""

        # 1. Metrics Comparison Radar Chart
        self._plot_radar_comparison(our_metrics, benchmark_metrics)

        # 2. Cumulative Returns Comparison
        self._plot_cumulative_returns(our_returns, benchmark_results)

        # 3. Risk-Return Scatter
        self._plot_risk_return_scatter(our_metrics, benchmark_metrics)

        # 4. Statistical Significance Heatmap
        self._plot_significance_heatmap(our_metrics, benchmark_metrics)

    def _plot_radar_comparison(
        self,
        our_metrics: StrategyMetrics,
        benchmark_metrics: Dict[str, StrategyMetrics]
    ):
        """Create radar chart comparing key metrics"""

        # Select key metrics for radar chart
        metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate', 'stability']
        labels = ['Sharpe', 'Sortino', 'Calmar', 'Win Rate', 'Stability']

        # Normalize metrics to 0-1 scale
        def normalize_metrics(metrics_dict):
            values = []
            for metric in metrics_to_plot:
                all_values = [getattr(our_metrics, metric)] + \
                            [getattr(m, metric) for m in benchmark_metrics.values()]
                max_val = max(all_values)
                min_val = min(all_values)
                range_val = max_val - min_val if max_val != min_val else 1
                norm_val = (metrics_dict[metric] - min_val) / range_val
                values.append(norm_val)
            return values

        our_values = normalize_metrics({m: getattr(our_metrics, m) for m in metrics_to_plot})

        # Compute average benchmark
        avg_bench_values = []
        for metric in metrics_to_plot:
            bench_vals = [getattr(m, metric) for m in benchmark_metrics.values()]
            avg_bench_values.append(np.mean(bench_vals))
        avg_bench_norm = normalize_metrics(dict(zip(metrics_to_plot, avg_bench_values)))

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        our_values += our_values[:1]
        avg_bench_norm += avg_bench_norm[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, our_values, 'o-', linewidth=2, label='Our RL Strategy', color='#06A77D')
        ax.fill(angles, our_values, alpha=0.25, color='#06A77D')
        ax.plot(angles, avg_bench_norm, 'o-', linewidth=2, label='Avg Benchmark', color='#D62828')
        ax.fill(angles, avg_bench_norm, alpha=0.25, color='#D62828')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=12)
        ax.set_ylim(0, 1)
        ax.set_title('Strategy Performance Comparison\n(Normalized Metrics)', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        save_path = self.output_dir / 'radar_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Radar chart saved: {save_path}")
        plt.close()

    def _plot_cumulative_returns(
        self,
        our_returns: np.ndarray,
        benchmark_results: Dict[str, BenchmarkResult]
    ):
        """Plot cumulative returns comparison"""

        fig, ax = plt.subplots(figsize=(15, 8))

        # Our strategy
        our_cumulative = np.cumprod(1 + our_returns)
        ax.plot(our_cumulative, linewidth=3, label='Our RL Strategy', color='#06A77D')

        # Benchmarks
        colors = plt.cm.tab10(np.linspace(0, 1, len(benchmark_results)))
        for (name, result), color in zip(benchmark_results.items(), colors):
            bench_cumulative = np.cumprod(1 + result.returns)
            min_len = min(len(our_cumulative), len(bench_cumulative))
            ax.plot(bench_cumulative[:min_len], linewidth=1.5, label=result.name, color=color, alpha=0.7)

        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.set_title('Cumulative Returns: Our Strategy vs Benchmarks', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / 'cumulative_returns.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Cumulative returns plot saved: {save_path}")
        plt.close()

    def _plot_risk_return_scatter(
        self,
        our_metrics: StrategyMetrics,
        benchmark_metrics: Dict[str, StrategyMetrics]
    ):
        """Create risk-return scatter plot"""

        fig, ax = plt.subplots(figsize=(12, 8))

        # Benchmarks
        for name, metrics in benchmark_metrics.items():
            ax.scatter(metrics.volatility, metrics.annualized_return,
                      s=150, alpha=0.6, label=name)

        # Our strategy (highlighted)
        ax.scatter(our_metrics.volatility, our_metrics.annualized_return,
                  s=300, alpha=1.0, color='#06A77D', edgecolors='black', linewidths=2,
                  marker='*', label='Our RL Strategy', zorder=10)

        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Annualized Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile: Our Strategy vs Benchmarks', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add Sharpe ratio contours
        vol_range = np.linspace(0, ax.get_xlim()[1], 100)
        for sharpe in [0.5, 1.0, 1.5, 2.0]:
            ret_range = sharpe * vol_range
            ax.plot(vol_range, ret_range, '--', alpha=0.3, color='gray')
            ax.text(vol_range[-1], ret_range[-1], f'Sharpe={sharpe}', fontsize=8, alpha=0.5)

        save_path = self.output_dir / 'risk_return_scatter.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Risk-return scatter saved: {save_path}")
        plt.close()

    def _plot_significance_heatmap(
        self,
        our_metrics: StrategyMetrics,
        benchmark_metrics: Dict[str, StrategyMetrics]
    ):
        """Create heatmap showing metric superiority"""

        # Metrics to compare
        metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return', 'max_drawdown', 'win_rate']
        strategies = ['Our Strategy'] + list(benchmark_metrics.keys())

        # Create comparison matrix
        matrix = []
        for strategy in strategies:
            row = []
            for metric in metrics:
                if strategy == 'Our Strategy':
                    value = getattr(our_metrics, metric)
                else:
                    value = getattr(benchmark_metrics[strategy], metric)

                # Normalize to percentage improvement over worst
                all_values = [getattr(our_metrics, metric)] + \
                            [getattr(m, metric) for m in benchmark_metrics.values()]

                if metric == 'max_drawdown':
                    # For drawdown, lower is better
                    worst = max(abs(v) for v in all_values)
                    normalized = (worst - abs(value)) / worst * 100 if worst != 0 else 0
                else:
                    # For others, higher is better
                    worst = min(all_values)
                    best = max(all_values)
                    range_val = best - worst if best != worst else 1
                    normalized = (value - worst) / range_val * 100

                row.append(normalized)

            matrix.append(row)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(strategies)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.set_yticklabels(strategies)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Score (0-100)', rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(strategies)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{matrix[i][j]:.0f}',
                             ha="center", va="center", color="black", fontsize=10)

        ax.set_title('Strategy Performance Heatmap\n(Higher is Better)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / 'performance_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance heatmap saved: {save_path}")
        plt.close()

    def _save_comparison_report(self, result: ComparisonResult):
        """Save detailed comparison report"""

        report = {
            'timestamp': datetime.now().isoformat(),
            'our_strategy': result.our_strategy.to_dict(),
            'benchmark_strategies': {
                name: metrics.to_dict()
                for name, metrics in result.benchmark_strategies.items()
            },
            'statistical_tests': result.statistical_tests,
            'rankings': result.rankings,
            'superiority_score': result.superiority_score
        }

        save_path = self.output_dir / 'comprehensive_comparison.json'
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Detailed comparison report saved: {save_path}")

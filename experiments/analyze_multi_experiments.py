"""
多次实验统计分析工具

用法：
    # 分析最近一次批量实验
    python analyze_multi_experiments.py

    # 分析指定批次
    python analyze_multi_experiments.py --results-dir outputs/multi_run_experiments/batch_20240115_120000

    # 与baseline比较
    python analyze_multi_experiments.py --compare-baseline

    # 生成完整报告
    python analyze_multi_experiments.py --full-report
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from evaluation.statistical_validation import StatisticalValidator


class MultiExperimentAnalyzer:
    """
    多次实验结果分析器

    功能：
    1. 加载多次运行的结果
    2. 进行统计显著性检验
    3. 计算效应量和统计功效
    4. 生成可视化报告
    """

    def __init__(self, results_dir: str):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing batch results
        """
        self.results_dir = Path(results_dir)

        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")

        print(f"\n{'='*80}")
        print(f"MULTI-EXPERIMENT STATISTICAL ANALYZER")
        print(f"{'='*80}")
        print(f"Results directory: {self.results_dir}")

        # Load batch summary
        summary_file = self.results_dir / 'batch_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                self.batch_summary = json.load(f)
            print(f"Batch ID: {self.batch_summary['batch_id']}")
            print(f"Total runs: {self.batch_summary['summary']['total_runs']}")
            print(f"Successful runs: {self.batch_summary['summary']['successful_runs']}")
        else:
            print(f"⚠️  No batch_summary.json found. Will scan directory.")
            self.batch_summary = None

        # Initialize validator
        self.validator = StatisticalValidator(
            output_dir=str(self.results_dir / "statistical_validation")
        )

        print(f"{'='*80}")

    def analyze(
        self,
        compare_baseline: bool = True,
        full_report: bool = True
    ) -> Dict:
        """
        Run complete statistical analysis.

        Args:
            compare_baseline: Compare with baseline strategy
            full_report: Generate full report with visualizations

        Returns:
            Analysis results
        """
        print(f"\n{'='*80}")
        print(f"LOADING EXPERIMENT RESULTS")
        print(f"{'='*80}")

        # Load all results
        results_df = self._load_results()

        if results_df is None or len(results_df) == 0:
            print(f"❌ No results found!")
            return {}

        print(f"Loaded {len(results_df)} successful runs")

        # Extract metrics
        metrics_to_analyze = [
            'sharpe_ratio',
            'total_return',
            'max_drawdown',
            'sortino_ratio',
            'calmar_ratio',
            'win_rate'
        ]

        results = {}

        # 1. Descriptive statistics
        print(f"\n{'='*80}")
        print(f"1. DESCRIPTIVE STATISTICS")
        print(f"{'='*80}")

        desc_stats = self._compute_descriptive_stats(results_df, metrics_to_analyze)
        results['descriptive_stats'] = desc_stats
        self._print_descriptive_stats(desc_stats)

        # 2. Compare with baseline (if available)
        if compare_baseline and 'baseline_sharpe' in results_df.columns:
            print(f"\n{'='*80}")
            print(f"2. STATISTICAL VALIDATION vs BASELINE")
            print(f"{'='*80}")

            validation_results = self._validate_against_baseline(results_df, metrics_to_analyze)
            results['validation_results'] = validation_results

        # 3. Sample size recommendations
        print(f"\n{'='*80}")
        print(f"3. SAMPLE SIZE RECOMMENDATIONS")
        print(f"{'='*80}")

        if 'baseline_sharpe' in results_df.columns:
            # Calculate observed effect size
            rl_sharpe = results_df['sharpe_ratio'].values
            baseline_sharpe = results_df['baseline_sharpe'].values
            differences = rl_sharpe - baseline_sharpe
            observed_effect_size = np.mean(differences) / np.std(differences, ddof=1)

            print(f"\nObserved effect size (Cohen's d): {observed_effect_size:.4f}")
            print(f"Current sample size: {len(results_df)}")

            # Compute required sample size for different power levels
            for power in [0.70, 0.80, 0.90]:
                required_n = self.validator.compute_required_sample_size(
                    expected_effect_size=observed_effect_size,
                    desired_power=power,
                    alpha=0.05
                )
                status = "✅" if len(results_df) >= required_n else "❌"
                print(f"\nFor {power*100:.0f}% power: {required_n} runs needed {status}")
                print(f"  Current: {len(results_df)} runs")

        # 4. Full report
        if full_report:
            print(f"\n{'='*80}")
            print(f"4. GENERATING COMPREHENSIVE REPORT")
            print(f"{'='*80}")

            self._generate_full_report(results_df, results)

        # Save analysis results
        self._save_analysis_results(results)

        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETED")
        print(f"{'='*80}")

        return results

    def _load_results(self) -> Optional[pd.DataFrame]:
        """Load all experiment results into DataFrame."""

        # Try to load from aggregated CSV first
        csv_file = self.results_dir / 'aggregated_metrics.csv'
        if csv_file.exists():
            print(f"Loading from aggregated CSV: {csv_file}")
            return pd.read_csv(csv_file)

        # Otherwise, load from batch summary
        if self.batch_summary:
            successful_results = [
                r for r in self.batch_summary['all_results']
                if r.get('status') == 'success' and r.get('sharpe_ratio') is not None
            ]

            if successful_results:
                return pd.DataFrame(successful_results)

        # Last resort: scan directory for individual result files
        print(f"Scanning directory for individual results...")
        all_results = []

        for run_dir in self.results_dir.glob("run_*"):
            if run_dir.is_dir():
                result_files = list(run_dir.glob("*_results.json"))
                if result_files:
                    with open(result_files[0], 'r') as f:
                        run_data = json.load(f)

                    if 'final_performance' in run_data:
                        metrics = run_data['final_performance'].copy()
                        metrics['seed'] = run_data.get('training_params', {}).get('seed', None)
                        all_results.append(metrics)

        if all_results:
            return pd.DataFrame(all_results)

        return None

    def _compute_descriptive_stats(
        self,
        df: pd.DataFrame,
        metrics: List[str]
    ) -> Dict:
        """Compute descriptive statistics."""
        stats = {}

        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna().values
                if len(values) > 0:
                    stats[metric] = {
                        'n': len(values),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values, ddof=1)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75))
                    }

        return stats

    def _print_descriptive_stats(self, stats: Dict):
        """Print descriptive statistics."""
        print(f"\n{'Metric':<20} {'N':>6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"{'─'*76}")

        for metric, s in stats.items():
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<20} {s['n']:>6} {s['mean']:>10.4f} {s['std']:>10.4f} "
                  f"{s['min']:>10.4f} {s['max']:>10.4f}")

    def _validate_against_baseline(
        self,
        df: pd.DataFrame,
        metrics: List[str]
    ) -> Dict:
        """Validate RL strategy against baseline."""
        validation_results = {}

        for metric in metrics:
            if metric in df.columns:
                rl_values = df[metric].dropna().values

                # Check if we have baseline values
                baseline_col = f'baseline_{metric}'
                if baseline_col not in df.columns:
                    # Try just 'baseline_sharpe' for sharpe_ratio
                    if metric == 'sharpe_ratio' and 'baseline_sharpe' in df.columns:
                        baseline_col = 'baseline_sharpe'
                    elif metric == 'total_return' and 'baseline_return' in df.columns:
                        baseline_col = 'baseline_return'
                    else:
                        continue

                baseline_values = df[baseline_col].dropna().values

                # Ensure same length
                min_len = min(len(rl_values), len(baseline_values))
                rl_values = rl_values[:min_len]
                baseline_values = baseline_values[:min_len]

                if len(rl_values) >= 10:  # Need at least 10 samples
                    print(f"\n{'─'*80}")
                    print(f"Testing: {metric.replace('_', ' ').title()}")
                    print(f"{'─'*80}")

                    result = self.validator.validate_superiority(
                        rl_metrics=rl_values,
                        baseline_metrics=baseline_values,
                        metric_name=metric.replace('_', ' ').title(),
                        alternative='greater'
                    )

                    validation_results[metric] = {
                        'p_value': result.p_value,
                        'effect_size': result.effect_size,
                        'power': result.power,
                        'is_significant': result.is_significant,
                        'confidence_interval': result.confidence_interval
                    }

                    # Bootstrap CI for robustness
                    if len(rl_values) >= 30:
                        mean_diff, ci_lower, ci_upper = self.validator.bootstrap_confidence_interval(
                            rl_metrics=rl_values,
                            baseline_metrics=baseline_values,
                            n_bootstrap=10000
                        )
                        validation_results[metric]['bootstrap_ci'] = (ci_lower, ci_upper)

        return validation_results

    def _generate_full_report(self, df: pd.DataFrame, analysis_results: Dict):
        """Generate comprehensive visualization report."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            sns.set_style("whitegrid")

            # Create multi-panel figure
            fig = plt.figure(figsize=(18, 12))

            # Define metrics to plot
            metrics_to_plot = [
                ('sharpe_ratio', 'Sharpe Ratio'),
                ('total_return', 'Total Return (%)'),
                ('max_drawdown', 'Max Drawdown (%)'),
                ('sortino_ratio', 'Sortino Ratio'),
                ('calmar_ratio', 'Calmar Ratio'),
                ('win_rate', 'Win Rate')
            ]

            for idx, (metric, label) in enumerate(metrics_to_plot, 1):
                if metric in df.columns:
                    ax = plt.subplot(2, 3, idx)

                    values = df[metric].dropna().values

                    # Histogram
                    ax.hist(values, bins=20, alpha=0.7, color='#06A77D', edgecolor='black')

                    # Mean line
                    mean_val = np.mean(values)
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                              label=f'Mean = {mean_val:.3f}')

                    # Baseline comparison if available
                    baseline_col = f'baseline_{metric}'
                    if baseline_col not in df.columns:
                        if metric == 'sharpe_ratio' and 'baseline_sharpe' in df.columns:
                            baseline_col = 'baseline_sharpe'
                        elif metric == 'total_return' and 'baseline_return' in df.columns:
                            baseline_col = 'baseline_return'

                    if baseline_col in df.columns:
                        baseline_mean = np.mean(df[baseline_col].dropna().values)
                        ax.axvline(baseline_mean, color='gray', linestyle=':', linewidth=2,
                                  label=f'Baseline = {baseline_mean:.3f}')

                    ax.set_xlabel(label, fontsize=11)
                    ax.set_ylabel('Frequency', fontsize=11)
                    ax.set_title(f'{label}\n(n={len(values)})', fontsize=11, fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            save_path = self.results_dir / 'comprehensive_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comprehensive report saved: {save_path}")
            plt.close()

            # Create comparison plot (RL vs Baseline)
            if 'baseline_sharpe' in df.columns:
                self._plot_rl_vs_baseline_comparison(df)

        except Exception as e:
            print(f"⚠️  Could not generate full report: {e}")

    def _plot_rl_vs_baseline_comparison(self, df: pd.DataFrame):
        """Create RL vs Baseline comparison plot."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 1: Paired comparison
            ax1 = axes[0]
            run_numbers = np.arange(1, len(df) + 1)
            ax1.plot(run_numbers, df['sharpe_ratio'].values, 'o-',
                    color='#06A77D', alpha=0.7, label='RL Strategy', linewidth=2, markersize=6)
            ax1.plot(run_numbers, df['baseline_sharpe'].values, 's-',
                    color='#D62828', alpha=0.7, label='Baseline', linewidth=2, markersize=6)

            ax1.axhline(np.mean(df['sharpe_ratio'].values), color='#06A77D',
                       linestyle='--', alpha=0.5, linewidth=1.5)
            ax1.axhline(np.mean(df['baseline_sharpe'].values), color='#D62828',
                       linestyle='--', alpha=0.5, linewidth=1.5)

            ax1.set_xlabel('Run Number', fontsize=12)
            ax1.set_ylabel('Sharpe Ratio', fontsize=12)
            ax1.set_title('Run-by-Run Comparison', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Box plot comparison
            ax2 = axes[1]
            data_to_plot = [df['baseline_sharpe'].values, df['sharpe_ratio'].values]
            bp = ax2.boxplot(data_to_plot, labels=['Baseline', 'RL Strategy'],
                            patch_artist=True, widths=0.6)
            bp['boxes'][0].set_facecolor('#D62828')
            bp['boxes'][1].set_facecolor('#06A77D')

            ax2.set_ylabel('Sharpe Ratio', fontsize=12)
            ax2.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Add significance indicator
            rl_mean = np.mean(df['sharpe_ratio'].values)
            baseline_mean = np.mean(df['baseline_sharpe'].values)
            improvement = ((rl_mean / baseline_mean) - 1) * 100

            ax2.text(0.5, 0.95, f'RL improves by {improvement:.1f}%',
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=11, fontweight='bold')

            plt.tight_layout()

            save_path = self.results_dir / 'rl_vs_baseline_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ RL vs Baseline comparison saved: {save_path}")
            plt.close()

        except Exception as e:
            print(f"⚠️  Could not generate comparison plot: {e}")

    def _save_analysis_results(self, results: Dict):
        """Save analysis results to JSON."""
        save_path = self.results_dir / 'statistical_analysis_results.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Analysis results saved: {save_path}")


def find_latest_batch(base_dir: str = "outputs/multi_run_experiments") -> Optional[Path]:
    """Find the most recent batch directory."""
    base_path = Path(base_dir)

    if not base_path.exists():
        return None

    batch_dirs = sorted(base_path.glob("batch_*"))

    if batch_dirs:
        return batch_dirs[-1]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-run experiment results with statistical validation"
    )

    parser.add_argument('--results-dir', type=str,
                       help='Directory containing batch results (auto-detect if not provided)')

    parser.add_argument('--compare-baseline', action='store_true', default=True,
                       help='Compare with baseline strategy (default: True)')

    parser.add_argument('--full-report', action='store_true', default=True,
                       help='Generate full report with visualizations (default: True)')

    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        print("No results directory specified. Looking for latest batch...")
        latest = find_latest_batch()
        if latest:
            results_dir = str(latest)
            print(f"Found: {results_dir}")
        else:
            print("❌ No batch directories found in outputs/multi_run_experiments/")
            print("Run experiments first: python run_multi_experiments.py")
            sys.exit(1)

    # Create analyzer
    try:
        analyzer = MultiExperimentAnalyzer(results_dir=results_dir)

        # Run analysis
        results = analyzer.analyze(
            compare_baseline=args.compare_baseline,
            full_report=args.full_report
        )

        print(f"\n✅ Analysis completed successfully!")

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
批量实验运行器 - 用于统计验证

用法：
    # 运行30次快速实验（最小可行样本量）
    python run_multi_experiments.py --n-runs 30 --episodes 100

    # 运行50次标准实验（推荐）
    python run_multi_experiments.py --n-runs 50 --episodes 500

    # 运行100次高标准实验（发表论文/商业应用）
    python run_multi_experiments.py --n-runs 100 --episodes 500 --regime high_risk

完成后：
    # 进行统计验证
    python analyze_multi_experiments.py --results-dir outputs/multi_run_experiments
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List
import pandas as pd


class BatchExperimentRunner:
    """
    批量实验运行器

    功能：
    1. 运行N次独立实验（不同随机种子）
    2. 收集所有运行的结果
    3. 保存汇总数据供统计分析
    """

    def __init__(
        self,
        n_runs: int,
        output_base_dir: str = "outputs/multi_run_experiments",
        seed_start: int = 1000
    ):
        """
        Initialize batch experiment runner.

        Args:
            n_runs: Number of independent runs
            output_base_dir: Base directory for all outputs
            seed_start: Starting seed number
        """
        self.n_runs = n_runs
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.seed_start = seed_start

        # Create timestamp for this batch
        self.batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_dir = self.output_base_dir / f"batch_{self.batch_id}"
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"BATCH EXPERIMENT RUNNER")
        print(f"{'='*80}")
        print(f"Number of runs: {n_runs}")
        print(f"Batch ID: {self.batch_id}")
        print(f"Output directory: {self.batch_dir}")
        print(f"Seed range: {seed_start} to {seed_start + n_runs - 1}")
        print(f"{'='*80}")

    def run_batch_experiments(
        self,
        mode: str = "all_regimes",
        data_source: str = "predefined",
        episodes: int = 500,
        steps: int = 100,
        gpu: bool = False,
        parallel: bool = False,
        max_parallel: int = 4
    ) -> Dict:
        """
        Run batch of experiments.

        Args:
            mode: Training mode ('auto', 'all_regimes', 'single')
            data_source: Data source
            episodes: Training episodes per run
            steps: Steps per episode
            gpu: Use GPU
            parallel: Run in parallel
            max_parallel: Max parallel processes

        Returns:
            Summary of all runs
        """
        print(f"\n{'='*80}")
        print(f"STARTING BATCH EXPERIMENTS")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Mode: {mode}")
        print(f"  Data source: {data_source}")
        print(f"  Episodes per run: {episodes}")
        print(f"  Steps per episode: {steps}")
        print(f"  GPU: {gpu}")
        print(f"  Parallel: {parallel} (max {max_parallel} processes)")
        print(f"{'='*80}")

        # Track all results
        all_results = []
        failed_runs = []
        successful_runs = 0

        if parallel:
            # TODO: Implement parallel execution
            print("\n⚠️  Parallel execution not yet implemented. Running sequentially.")
            parallel = False

        # Run experiments sequentially
        for run_idx in range(self.n_runs):
            seed = self.seed_start + run_idx
            run_name = f"run_{run_idx+1:03d}_seed_{seed}"

            print(f"\n{'─'*80}")
            print(f"RUN {run_idx + 1}/{self.n_runs}: {run_name}")
            print(f"{'─'*80}")

            try:
                # Create output directory for this run
                run_output_dir = self.batch_dir / run_name
                run_output_dir.mkdir(parents=True, exist_ok=True)

                # Build command
                cmd = [
                    sys.executable, "train.py",
                    "--mode", mode,
                    "--data-source", data_source,
                    "--episodes", str(episodes),
                    "--steps", str(steps),
                    "--seed", str(seed),
                    "--output", str(run_output_dir)
                ]

                if gpu:
                    cmd.append("--gpu")

                print(f"Command: {' '.join(cmd)}")
                print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

                # Run training
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )

                if result.returncode == 0:
                    print(f"✅ Run {run_idx + 1} COMPLETED")
                    successful_runs += 1

                    # Extract results
                    run_results = self._extract_run_results(run_output_dir)
                    run_results['run_idx'] = run_idx + 1
                    run_results['seed'] = seed
                    run_results['status'] = 'success'
                    all_results.append(run_results)

                else:
                    print(f"❌ Run {run_idx + 1} FAILED")
                    print(f"Return code: {result.returncode}")
                    if result.stderr:
                        print(f"Error: {result.stderr[:500]}")  # First 500 chars

                    failed_runs.append({
                        'run_idx': run_idx + 1,
                        'seed': seed,
                        'error': result.stderr[:500] if result.stderr else "Unknown error"
                    })

                    # Still record as failed
                    all_results.append({
                        'run_idx': run_idx + 1,
                        'seed': seed,
                        'status': 'failed',
                        'error': result.stderr[:500] if result.stderr else "Unknown error"
                    })

                print(f"Completed at: {datetime.now().strftime('%H:%M:%S')}")

            except subprocess.TimeoutExpired:
                print(f"❌ Run {run_idx + 1} TIMEOUT (>1 hour)")
                failed_runs.append({
                    'run_idx': run_idx + 1,
                    'seed': seed,
                    'error': 'Timeout after 1 hour'
                })
                all_results.append({
                    'run_idx': run_idx + 1,
                    'seed': seed,
                    'status': 'timeout'
                })

            except Exception as e:
                print(f"❌ Run {run_idx + 1} EXCEPTION: {e}")
                failed_runs.append({
                    'run_idx': run_idx + 1,
                    'seed': seed,
                    'error': str(e)
                })
                all_results.append({
                    'run_idx': run_idx + 1,
                    'seed': seed,
                    'status': 'exception',
                    'error': str(e)
                })

            # Print progress
            progress_pct = ((run_idx + 1) / self.n_runs) * 100
            print(f"\nProgress: {run_idx + 1}/{self.n_runs} ({progress_pct:.1f}%)")
            print(f"Successful: {successful_runs}, Failed: {len(failed_runs)}")

        # Compile summary
        summary = {
            'batch_id': self.batch_id,
            'config': {
                'n_runs': self.n_runs,
                'mode': mode,
                'data_source': data_source,
                'episodes': episodes,
                'steps': steps,
                'seed_start': self.seed_start,
                'seed_end': self.seed_start + self.n_runs - 1
            },
            'summary': {
                'total_runs': self.n_runs,
                'successful_runs': successful_runs,
                'failed_runs': len(failed_runs),
                'success_rate': successful_runs / self.n_runs * 100
            },
            'all_results': all_results,
            'failed_runs': failed_runs,
            'timestamp': datetime.now().isoformat()
        }

        # Save summary
        self._save_summary(summary)

        # Print final summary
        self._print_final_summary(summary)

        # Generate aggregated metrics
        if successful_runs >= 10:
            self._generate_aggregated_metrics(all_results)

        return summary

    def _extract_run_results(self, run_dir: Path) -> Dict:
        """Extract results from a single run."""
        results = {}

        # Look for result JSON files
        json_files = list(run_dir.glob("*_results.json"))

        if json_files:
            # Load the first results file found
            with open(json_files[0], 'r') as f:
                run_data = json.load(f)

            # Extract key metrics
            if 'final_performance' in run_data:
                perf = run_data['final_performance']
                results.update({
                    'sharpe_ratio': perf.get('sharpe_ratio', None),
                    'total_return': perf.get('total_return', None),
                    'max_drawdown': perf.get('max_drawdown', None),
                    'sortino_ratio': perf.get('sortino_ratio', None),
                    'calmar_ratio': perf.get('calmar_ratio', None),
                    'win_rate': perf.get('win_rate', None),
                    'volatility': perf.get('volatility', None),
                    'var_95': perf.get('var_95', None),
                    'cvar_95': perf.get('cvar_95', None)
                })

            # Extract baseline stats
            if 'baseline_stats' in run_data:
                baseline = run_data['baseline_stats']
                results['baseline_sharpe'] = baseline.get('sharpe_ratio', None)
                results['baseline_return'] = baseline.get('total_return', None)

            # Extract regime info
            if 'regime_type' in run_data:
                results['regime_type'] = run_data['regime_type']

        return results

    def _save_summary(self, summary: Dict):
        """Save batch summary."""
        # Save full summary
        summary_file = self.batch_dir / 'batch_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Batch summary saved: {summary_file}")

        # Save metrics only (for easy loading)
        successful_results = [r for r in summary['all_results'] if r.get('status') == 'success']

        if successful_results:
            metrics_df = pd.DataFrame(successful_results)
            metrics_csv = self.batch_dir / 'aggregated_metrics.csv'
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"💾 Aggregated metrics saved: {metrics_csv}")

    def _print_final_summary(self, summary: Dict):
        """Print final summary."""
        print(f"\n{'='*80}")
        print(f"BATCH EXPERIMENTS COMPLETED")
        print(f"{'='*80}")

        s = summary['summary']
        print(f"\n📊 SUMMARY")
        print(f"{'─'*80}")
        print(f"Total runs:       {s['total_runs']}")
        print(f"Successful runs:  {s['successful_runs']} ({s['success_rate']:.1f}%)")
        print(f"Failed runs:      {s['failed_runs']}")

        # Extract metrics from successful runs
        successful_results = [r for r in summary['all_results']
                            if r.get('status') == 'success' and r.get('sharpe_ratio') is not None]

        if successful_results:
            sharpe_ratios = [r['sharpe_ratio'] for r in successful_results]
            total_returns = [r['total_return'] for r in successful_results]
            max_drawdowns = [r['max_drawdown'] for r in successful_results]

            print(f"\n📈 PERFORMANCE METRICS (Mean ± Std)")
            print(f"{'─'*80}")
            print(f"Sharpe Ratio:     {np.mean(sharpe_ratios):.4f} ± {np.std(sharpe_ratios):.4f}")
            print(f"Total Return:     {np.mean(total_returns):.2f}% ± {np.std(total_returns):.2f}%")
            print(f"Max Drawdown:     {np.mean(max_drawdowns):.2f}% ± {np.std(max_drawdowns):.2f}%")

            # Compare with baseline if available
            if all('baseline_sharpe' in r for r in successful_results):
                baseline_sharpes = [r['baseline_sharpe'] for r in successful_results]
                print(f"\nBaseline Sharpe:  {np.mean(baseline_sharpes):.4f} ± {np.std(baseline_sharpes):.4f}")
                print(f"\nImprovement:      {np.mean(sharpe_ratios) - np.mean(baseline_sharpes):.4f} "
                      f"({((np.mean(sharpe_ratios) / np.mean(baseline_sharpes)) - 1) * 100:.1f}%)")

        print(f"\n💡 NEXT STEPS")
        print(f"{'─'*80}")
        print(f"1. Run statistical validation:")
        print(f"   python analyze_multi_experiments.py --results-dir {self.batch_dir}")
        print(f"\n2. Check individual runs:")
        print(f"   ls {self.batch_dir}/run_*")
        print(f"\n3. View aggregated metrics:")
        print(f"   cat {self.batch_dir}/aggregated_metrics.csv")
        print(f"{'='*80}")

    def _generate_aggregated_metrics(self, all_results: List[Dict]):
        """Generate aggregated metrics visualization."""
        successful_results = [r for r in all_results if r.get('status') == 'success']

        if len(successful_results) < 10:
            print(f"\n⚠️  Only {len(successful_results)} successful runs. Need at least 10 for visualization.")
            return

        print(f"\n📊 Generating aggregated metrics visualization...")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Extract metrics
            metrics = {
                'Sharpe Ratio': [r.get('sharpe_ratio') for r in successful_results if r.get('sharpe_ratio') is not None],
                'Total Return (%)': [r.get('total_return') for r in successful_results if r.get('total_return') is not None],
                'Max Drawdown (%)': [abs(r.get('max_drawdown', 0)) for r in successful_results if r.get('max_drawdown') is not None],
            }

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for idx, (metric_name, values) in enumerate(metrics.items()):
                if values:
                    ax = axes[idx]
                    ax.hist(values, bins=20, alpha=0.7, color='#06A77D', edgecolor='black')
                    ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2,
                              label=f'Mean = {np.mean(values):.3f}')
                    ax.set_xlabel(metric_name, fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.set_title(f'{metric_name}\n(n={len(values)})', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.batch_dir / 'metrics_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")
            plt.close()

        except Exception as e:
            print(f"⚠️  Could not generate visualization: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple independent experiments for statistical validation"
    )

    parser.add_argument('--n-runs', type=int, default=30,
                       help='Number of independent runs (default: 30, recommended: 50-100)')

    parser.add_argument('--mode', type=str, default='all_regimes',
                       choices=['auto', 'all_regimes', 'single'],
                       help='Training mode')

    parser.add_argument('--data-source', type=str, default='predefined',
                       choices=['yahoo', 'csv', 'predefined'],
                       help='Data source')

    parser.add_argument('--episodes', type=int, default=500,
                       help='Training episodes per run')

    parser.add_argument('--steps', type=int, default=100,
                       help='Steps per episode')

    parser.add_argument('--seed-start', type=int, default=1000,
                       help='Starting seed number')

    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')

    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel (experimental)')

    parser.add_argument('--max-parallel', type=int, default=4,
                       help='Max parallel processes')

    parser.add_argument('--output', type=str, default='outputs/multi_run_experiments',
                       help='Base output directory')

    args = parser.parse_args()

    # Validate n_runs
    if args.n_runs < 10:
        print(f"⚠️  WARNING: Only {args.n_runs} runs requested.")
        print(f"   For reliable statistical validation, recommend at least 30 runs.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Create runner
    runner = BatchExperimentRunner(
        n_runs=args.n_runs,
        output_base_dir=args.output,
        seed_start=args.seed_start
    )

    # Run batch
    summary = runner.run_batch_experiments(
        mode=args.mode,
        data_source=args.data_source,
        episodes=args.episodes,
        steps=args.steps,
        gpu=args.gpu,
        parallel=args.parallel,
        max_parallel=args.max_parallel
    )

    print(f"\n✅ All experiments completed!")
    print(f"Results directory: {runner.batch_dir}")


if __name__ == "__main__":
    main()

"""
Lightweight Experiment Script - Data Analysis Only

This script is for quick data analysis and regime detection WITHOUT training.
For actual trading system training on real data, use: train.py

This script demonstrates:
1. Loading real market data
2. Detecting market regimes (high risk, high return, stable periods)
3. Computing baseline statistics
4. NO actual agent training

Usage:
    # Quick analysis of predefined periods
    python run_experiments.py --experiment predefined

    # Analyze custom period
    python run_experiments.py --experiment custom --symbols AAPL MSFT --start 2020-01-01 --end 2021-12-31

    # Find best periods for each regime
    python run_experiments.py --experiment sensitivity --symbols ^GSPC --start 2019-01-01 --end 2022-12-31

For TRAINING with real data, use:
    python train.py --mode all_regimes --episodes 500
"""

import argparse
from experiments import ExperimentRunner

print("\n" + "=" * 80)
print("NOTE: This is a lightweight analysis script (NO training)")
print("For actual trading system training, use: train.py")
print("=" * 80)


def run_predefined_experiments():
    """Run experiments on predefined market periods."""
    runner = ExperimentRunner(output_dir="outputs/experiments")

    print("\n" + "=" * 80)
    print("RUNNING PREDEFINED EXPERIMENTS")
    print("=" * 80)
    print("\nThis will test the trading system on three distinct market conditions:")
    print("  1. HIGH RISK PERIOD (超大风险期): COVID-19 Crash (Feb-Apr 2020)")
    print("  2. HIGH RETURN PERIOD (超大收益期): Post-COVID Rally (May 2020 - Dec 2021)")
    print("  3. STABLE PERIOD (平稳期): Pre-COVID Market (2019)")
    print("=" * 80)

    # Run experiments
    results = runner.run_predefined_experiments()

    return results


def run_custom_experiment(symbols, start_date, end_date):
    """Run experiment on custom data."""
    runner = ExperimentRunner(output_dir="outputs/experiments")

    print("\n" + "=" * 80)
    print("RUNNING CUSTOM EXPERIMENT")
    print("=" * 80)
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 80)

    results = runner.run_custom_experiment(
        data_source='yahoo',
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )

    return results


def run_sensitivity_analysis(symbols, start_date, end_date):
    """Run regime sensitivity analysis."""
    runner = ExperimentRunner(output_dir="outputs/experiments")

    print("\n" + "=" * 80)
    print("REGIME SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 80)
    print("\nThis will:")
    print("  1. Detect all market regimes in the specified period")
    print("  2. Identify the top periods for each regime type")
    print("  3. Analyze strategy performance in each regime")
    print("=" * 80)

    results = runner.analyze_regime_sensitivity(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run trading system experiments on different market regimes"
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default='predefined',
        choices=['predefined', 'custom', 'sensitivity'],
        help='Type of experiment to run'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['^GSPC', 'AAPL', 'MSFT'],
        help='Symbols to analyze (for custom/sensitivity experiments)'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2019-01-01',
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2022-12-31',
        help='End date (YYYY-MM-DD)'
    )

    args = parser.parse_args()

    # Run selected experiment
    if args.experiment == 'predefined':
        results = run_predefined_experiments()

    elif args.experiment == 'custom':
        results = run_custom_experiment(
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end
        )

    elif args.experiment == 'sensitivity':
        results = run_sensitivity_analysis(
            symbols=args.symbols,
            start_date=args.start,
            end_date=args.end
        )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print("\nResults saved to: outputs/experiments/")
    print("\nNext steps:")
    print("  1. Review the experiment results in outputs/experiments/")
    print("  2. Compare performance across different market regimes")
    print("  3. Integrate insights into your trading strategy")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Experiment Runner

Runs experiments across different market regimes (high risk, high return, stable).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimePeriod
from .real_data_loader import RealDataLoader
from .strategy_monitor import StrategyMonitor


class ExperimentRunner:
    """
    Runner for experiments across different market regimes.

    Manages:
    1. Data loading and regime detection
    2. Experiment execution for each regime
    3. Performance tracking and comparison
    """

    def __init__(
        self,
        output_dir: str = "outputs/experiments",
        verbose: bool = True
    ):
        """
        Initialize the experiment runner.

        Args:
            output_dir: Directory to save experiment results
            verbose: Whether to print detailed logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Initialize components
        self.data_loader = RealDataLoader(verbose=verbose)
        self.regime_detector = MarketRegimeDetector()

        # Results storage
        self.experiment_results = {}

        if self.verbose:
            print(f"Experiment Runner initialized. Output: {output_dir}")

    def run_predefined_experiments(
        self,
        experiment_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run experiments on predefined market periods:
        1. High Risk Period (COVID-19 crash)
        2. High Return Period (Post-COVID rally)
        3. Stable Period (2019)

        Args:
            experiment_config: Optional configuration for experiments

        Returns:
            Dictionary with experiment results
        """
        if experiment_config is None:
            experiment_config = self._get_default_config()

        results = {
            'high_risk': None,
            'high_return': None,
            'stable': None,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "=" * 80)
        print("RUNNING PREDEFINED EXPERIMENTS")
        print("=" * 80)

        # Experiment 1: High Risk Period (COVID-19 crash)
        print("\n[1/3] High Risk Period - COVID-19 Crash (Feb-Apr 2020)")
        print("-" * 80)
        try:
            crisis_data = self.data_loader.get_crisis_data('covid_2020')
            results['high_risk'] = self._run_experiment_on_data(
                data=crisis_data,
                regime_type='high_risk',
                config=experiment_config
            )
            print("✓ High risk experiment completed")
        except Exception as e:
            print(f"✗ High risk experiment failed: {e}")
            results['high_risk'] = {'error': str(e)}

        # Experiment 2: High Return Period (Post-COVID rally)
        print("\n[2/3] High Return Period - Post-COVID Rally (May 2020 - Dec 2021)")
        print("-" * 80)
        try:
            bull_data = self.data_loader.get_bull_market_data('post_covid_2020')
            results['high_return'] = self._run_experiment_on_data(
                data=bull_data,
                regime_type='high_return',
                config=experiment_config
            )
            print("✓ High return experiment completed")
        except Exception as e:
            print(f"✗ High return experiment failed: {e}")
            results['high_return'] = {'error': str(e)}

        # Experiment 3: Stable Period (2019)
        print("\n[3/3] Stable Period - Pre-COVID Market (2019)")
        print("-" * 80)
        try:
            stable_data = self.data_loader.get_stable_market_data(year=2019)
            results['stable'] = self._run_experiment_on_data(
                data=stable_data,
                regime_type='stable',
                config=experiment_config
            )
            print("✓ Stable period experiment completed")
        except Exception as e:
            print(f"✗ Stable period experiment failed: {e}")
            results['stable'] = {'error': str(e)}

        # Save results
        self._save_results(results)

        # Print comparison
        self._print_comparison(results)

        return results

    def run_custom_experiment(
        self,
        data_source: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run experiment on custom data.

        Args:
            data_source: Data source ('yahoo', 'csv')
            symbols: List of symbols or file paths
            start_date: Start date
            end_date: End date
            config: Experiment configuration

        Returns:
            Dictionary with experiment results
        """
        if config is None:
            config = self._get_default_config()

        print("\n" + "=" * 80)
        print(f"RUNNING CUSTOM EXPERIMENT: {symbols}")
        print(f"Period: {start_date} to {end_date}")
        print("=" * 80)

        # Load data
        if data_source == 'yahoo':
            data = self.data_loader.load_multiple_assets(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

        # Detect regimes
        main_symbol = symbols[0]
        prices = data[main_symbol]['Close'].values
        detected_periods = self.regime_detector.detect_specific_periods(
            prices=prices,
            dates=data[main_symbol].index.to_pydatetime().tolist()
        )

        print(f"\nDetected regimes:")
        for regime, periods in detected_periods.items():
            print(f"  {regime}: {len(periods)} periods")

        # Run experiments for each detected regime
        results = {}
        for regime_type, periods in detected_periods.items():
            if not periods:
                continue

            print(f"\nRunning experiment for {regime_type}...")

            # Use the longest period
            longest_period = max(periods, key=lambda p: p.end_idx - p.start_idx)

            # Extract data for this period
            period_data = {}
            for symbol, df in data.items():
                period_df = df.iloc[longest_period.start_idx:longest_period.end_idx]
                period_data[symbol] = period_df

            results[regime_type] = self._run_experiment_on_data(
                data=period_data,
                regime_type=regime_type,
                config=config
            )

        self._save_results(results, filename='custom_experiment_results.json')

        return results

    def _run_experiment_on_data(
        self,
        data: Dict[str, pd.DataFrame],
        regime_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run experiment on given data.

        This is a placeholder that would integrate with your actual trading system.
        In the full implementation, this would:
        1. Initialize agents with the data
        2. Run training/evaluation
        3. Collect performance metrics

        Args:
            data: Market data
            regime_type: Type of market regime
            config: Experiment configuration

        Returns:
            Dictionary with experiment results
        """
        # Extract main asset data
        main_symbol = list(data.keys())[0]
        df = data[main_symbol]

        # Compute basic statistics
        prices = df['Close'].values
        returns = np.diff(prices) / prices[:-1]

        stats = {
            'regime_type': regime_type,
            'num_periods': len(df),
            'num_assets': len(data),
            'assets': list(data.keys()),
            'start_date': df.index[0].isoformat() if isinstance(df.index[0], pd.Timestamp) else str(df.index[0]),
            'end_date': df.index[-1].isoformat() if isinstance(df.index[-1], pd.Timestamp) else str(df.index[-1]),
            'performance': {
                'total_return': float((prices[-1] / prices[0] - 1) * 100),
                'volatility': float(np.std(returns) * np.sqrt(252) * 100),
                'sharpe_ratio': float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0,
                'max_drawdown': float(self._compute_max_drawdown(prices) * 100),
                'win_rate': float(np.sum(returns > 0) / len(returns) * 100),
            }
        }

        # Prepare data for different agents
        if self.verbose:
            print(f"\n  Statistics for {regime_type}:")
            print(f"    Total Return: {stats['performance']['total_return']:.2f}%")
            print(f"    Volatility: {stats['performance']['volatility']:.2f}%")
            print(f"    Sharpe Ratio: {stats['performance']['sharpe_ratio']:.2f}")
            print(f"    Max Drawdown: {stats['performance']['max_drawdown']:.2f}%")

        # Placeholder for actual agent training/evaluation
        # In full implementation, you would:
        # 1. Initialize your MultiFrequencySystem with this data
        # 2. Run training or evaluation
        # 3. Collect detailed metrics

        stats['note'] = "This is a data analysis. Integrate with train_full_system.py for actual agent training."

        return stats

    def _compute_max_drawdown(self, prices: np.ndarray) -> float:
        """Compute maximum drawdown."""
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        return float(np.min(drawdown))

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default experiment configuration."""
        return {
            'num_episodes': 100,
            'steps_per_episode': 100,
            'learning_rate': 3e-4,
            'batch_size': 256,
            'buffer_size': 100000,
        }

    def _save_results(self, results: Dict[str, Any], filename: str = 'experiment_results.json'):
        """Save experiment results to file."""
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\n✓ Results saved to: {filepath}")

    def _print_comparison(self, results: Dict[str, Any]):
        """Print comparison of results across regimes."""
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPARISON")
        print("=" * 80)

        regimes = ['high_risk', 'high_return', 'stable']

        # Print table header
        print(f"\n{'Metric':<25} {'High Risk':>15} {'High Return':>15} {'Stable':>15}")
        print("-" * 80)

        # Extract metrics
        metrics_to_compare = [
            ('Total Return (%)', 'total_return'),
            ('Volatility (%)', 'volatility'),
            ('Sharpe Ratio', 'sharpe_ratio'),
            ('Max Drawdown (%)', 'max_drawdown'),
            ('Win Rate (%)', 'win_rate')
        ]

        for metric_name, metric_key in metrics_to_compare:
            values = []
            for regime in regimes:
                if results.get(regime) and 'performance' in results[regime]:
                    value = results[regime]['performance'].get(metric_key, 0.0)
                    values.append(f"{value:>15.2f}")
                else:
                    values.append(f"{'N/A':>15}")

            print(f"{metric_name:<25} {values[0]} {values[1]} {values[2]}")

        print("=" * 80)

    def analyze_regime_sensitivity(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Analyze how strategies perform across different detected regimes.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date

        Returns:
            Analysis results
        """
        print("\n" + "=" * 80)
        print("REGIME SENSITIVITY ANALYSIS")
        print("=" * 80)

        # Load data
        data = self.data_loader.load_multiple_assets(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )

        main_symbol = symbols[0]
        prices = data[main_symbol]['Close'].values
        dates = data[main_symbol].index.to_pydatetime().tolist()

        # Detect all regimes
        detected_periods = self.regime_detector.detect_specific_periods(
            prices=prices,
            dates=dates,
            min_period_length=20
        )

        # Get top periods for each regime
        top_periods = self.regime_detector.get_top_periods(detected_periods, n=3)

        # Print summary
        self.regime_detector.print_period_summary(top_periods)

        # Analyze each top period
        analysis_results = {}
        for regime_type, periods in top_periods.items():
            analysis_results[regime_type] = []

            for i, period in enumerate(periods):
                # Extract data for this period
                period_data = {}
                for symbol, df in data.items():
                    period_df = df.iloc[period.start_idx:period.end_idx]
                    period_data[symbol] = period_df

                # Run analysis
                result = self._run_experiment_on_data(
                    data=period_data,
                    regime_type=regime_type,
                    config=self._get_default_config()
                )

                result['period_info'] = {
                    'start_date': period.start_date.isoformat() if period.start_date else None,
                    'end_date': period.end_date.isoformat() if period.end_date else None,
                    'volatility': period.volatility,
                    'avg_return': period.avg_return,
                    'sharpe_ratio': period.sharpe_ratio,
                    'max_drawdown': period.max_drawdown
                }

                analysis_results[regime_type].append(result)

        # Save analysis
        self._save_results(analysis_results, filename='regime_sensitivity_analysis.json')

        return analysis_results


def main():
    """Example usage of ExperimentRunner."""
    runner = ExperimentRunner(output_dir="outputs/experiments")

    # Run predefined experiments
    print("\nRunning predefined experiments...")
    results = runner.run_predefined_experiments()

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()

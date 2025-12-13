"""
Visualization Tools for Strategy Evaluation

Comprehensive visualization for trading strategy performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class EvaluationVisualizer:
    """Visualization tools for strategy evaluation"""

    def __init__(self, output_dir: str = "outputs/evaluation"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_comprehensive_report(
        self,
        returns: np.ndarray,
        capital_series: np.ndarray,
        metrics: Dict,
        benchmark_returns: Optional[np.ndarray] = None,
        benchmark_capital: Optional[np.ndarray] = None,
        regime: Optional[str] = None,
        save_name: str = "strategy_evaluation.png"
    ):
        """
        Create comprehensive evaluation report with multiple plots.

        Args:
            returns: Strategy returns
            capital_series: Capital over time
            metrics: Performance metrics dictionary
            benchmark_returns: Optional benchmark returns
            benchmark_capital: Optional benchmark capital
            regime: Market regime name
            save_name: Filename to save plot
        """
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        title = f"Strategy Evaluation Report"
        if regime:
            title += f" - {regime.upper()} Regime"
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_cumulative_returns(
            ax1, capital_series, benchmark_capital, "Cumulative Capital"
        )

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_drawdown(ax2, capital_series)

        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax3, returns)

        # 4. Rolling Sharpe
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_rolling_sharpe(ax4, returns)

        # 5. Monthly Returns Heatmap
        ax5 = fig.add_subplot(gs[2, 2])
        self._plot_monthly_heatmap(ax5, returns)

        # 6. Metrics Summary Table
        ax6 = fig.add_subplot(gs[0:2, 2])
        self._plot_metrics_table(ax6, metrics)

        # Save
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Evaluation report saved: {save_path}")

    def _plot_cumulative_returns(
        self,
        ax,
        capital: np.ndarray,
        benchmark_capital: Optional[np.ndarray],
        title: str
    ):
        """Plot cumulative returns"""
        # Normalize to start at 1
        normalized = capital / capital[0]

        ax.plot(normalized, label='Strategy', linewidth=2, color='#2E86AB')

        if benchmark_capital is not None:
            benchmark_norm = benchmark_capital / benchmark_capital[0]
            ax.plot(benchmark_norm, label='Benchmark (Buy & Hold)',
                   linewidth=2, color='#A23B72', linestyle='--', alpha=0.7)

        ax.set_xlabel('Period')
        ax.set_ylabel('Cumulative Return (Normalized)')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Shade positive/negative regions
        ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.fill_between(range(len(normalized)), 1, normalized,
                        where=(normalized >= 1), alpha=0.1, color='green', label='_nolegend_')
        ax.fill_between(range(len(normalized)), 1, normalized,
                        where=(normalized < 1), alpha=0.1, color='red', label='_nolegend_')

    def _plot_drawdown(self, ax, capital: np.ndarray):
        """Plot drawdown over time"""
        cummax = np.maximum.accumulate(capital)
        drawdown = (capital - cummax) / cummax * 100

        ax.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        ax.plot(drawdown, color='darkred', linewidth=1.5)

        ax.set_xlabel('Period')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Over Time', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Highlight max drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd_val = drawdown[max_dd_idx]
        ax.scatter([max_dd_idx], [max_dd_val], color='darkred', s=100,
                  zorder=5, label=f'Max DD: {max_dd_val:.2f}%')
        ax.legend(loc='lower left')

    def _plot_returns_distribution(self, ax, returns: np.ndarray):
        """Plot returns distribution with normal curve"""
        # Histogram
        n, bins, patches = ax.hist(returns * 100, bins=50, density=True,
                                   alpha=0.7, color='skyblue', edgecolor='black')

        # Fit normal distribution
        mu, sigma = np.mean(returns * 100), np.std(returns * 100)
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2)),
               linewidth=2, color='red', label='Normal Distribution')

        # Add vertical lines for mean and median
        ax.axvline(mu, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mu:.3f}%')
        ax.axvline(np.median(returns * 100), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(returns * 100):.3f}%')

        ax.set_xlabel('Returns (%)')
        ax.set_ylabel('Density')
        ax.set_title('Returns Distribution', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_rolling_sharpe(self, ax, returns: np.ndarray, window: int = 50):
        """Plot rolling Sharpe ratio"""
        if len(returns) < window:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Rolling Sharpe Ratio', fontweight='bold')
            return

        rolling_mean = np.array([np.mean(returns[max(0, i-window):i+1])
                                for i in range(len(returns))])
        rolling_std = np.array([np.std(returns[max(0, i-window):i+1])
                               for i in range(len(returns))])

        rolling_sharpe = np.where(rolling_std > 0,
                                 rolling_mean / rolling_std * np.sqrt(252),
                                 0)

        ax.plot(rolling_sharpe, linewidth=1.5, color='purple')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')
        ax.axhline(y=2, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 2')

        ax.set_xlabel('Period')
        ax.set_ylabel('Rolling Sharpe Ratio')
        ax.set_title(f'Rolling Sharpe Ratio (Window={window})', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_monthly_heatmap(self, ax, returns: np.ndarray):
        """Plot monthly returns heatmap (simulated)"""
        # Simulate monthly data from daily/period data
        # For actual implementation, would need date information
        n_months = max(1, len(returns) // 21)  # Approximate months
        monthly_returns = []

        for i in range(n_months):
            start_idx = i * 21
            end_idx = min((i + 1) * 21, len(returns))
            if start_idx < len(returns):
                month_ret = np.prod(1 + returns[start_idx:end_idx]) - 1
                monthly_returns.append(month_ret * 100)

        if len(monthly_returns) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Monthly Returns', fontweight='bold')
            return

        # Reshape into years x months
        years = (len(monthly_returns) + 11) // 12
        data = np.zeros((years, 12))
        data[:] = np.nan

        for i, ret in enumerate(monthly_returns):
            year_idx = i // 12
            month_idx = i % 12
            if year_idx < years:
                data[year_idx, month_idx] = ret

        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        ax.set_xticks(range(12))
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.set_yticks(range(years))
        ax.set_yticklabels([f'Y{i+1}' for i in range(years)])
        ax.set_title('Monthly Returns Heatmap (%)', fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(years):
            for j in range(12):
                if not np.isnan(data[i, j]):
                    text = ax.text(j, i, f'{data[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=7)

    def _plot_metrics_table(self, ax, metrics: Dict):
        """Plot metrics as a formatted table"""
        ax.axis('off')

        # Organize metrics into categories
        categories = {
            'Return Metrics': [
                ('Total Return', metrics.get('total_return', 0), '%'),
                ('Annualized Return', metrics.get('annualized_return', 0), '%'),
                ('Avg Return', metrics.get('avg_return', 0) * 100, '%'),
            ],
            'Risk Metrics': [
                ('Volatility', metrics.get('volatility', 0), '%'),
                ('Max Drawdown', metrics.get('max_drawdown', 0), '%'),
                ('Downside Vol.', metrics.get('downside_volatility', 0), '%'),
            ],
            'Risk-Adjusted': [
                ('Sharpe Ratio', metrics.get('sharpe_ratio', 0), ''),
                ('Sortino Ratio', metrics.get('sortino_ratio', 0), ''),
                ('Calmar Ratio', metrics.get('calmar_ratio', 0), ''),
            ],
            'Risk Measures': [
                ('VaR 95%', metrics.get('var_95', 0), '%'),
                ('CVaR 95%', metrics.get('cvar_95', 0), '%'),
                ('VaR 99%', metrics.get('var_99', 0), '%'),
            ],
            'Trading Metrics': [
                ('Win Rate', metrics.get('win_rate', 0) * 100, '%'),
                ('Profit Factor', metrics.get('profit_factor', 0), ''),
                ('Num Trades', int(metrics.get('num_trades', 0)), ''),
            ]
        }

        # Create table text
        y_pos = 0.95
        for category, items in categories.items():
            # Category header
            ax.text(0.05, y_pos, category, fontsize=11, fontweight='bold',
                   transform=ax.transAxes)
            y_pos -= 0.04

            # Items
            for name, value, unit in items:
                if isinstance(value, int):
                    text = f"{name}: {value}{unit}"
                else:
                    text = f"{name}: {value:.2f}{unit}"

                # Color code based on value
                color = 'black'
                if 'Return' in name and value > 0:
                    color = 'green'
                elif 'Return' in name and value < 0:
                    color = 'red'
                elif 'Sharpe' in name or 'Sortino' in name or 'Calmar' in name:
                    if value > 1:
                        color = 'green'
                    elif value < 0:
                        color = 'red'

                ax.text(0.1, y_pos, text, fontsize=9, color=color,
                       transform=ax.transAxes)
                y_pos -= 0.03

            y_pos -= 0.02

    def plot_regime_comparison(
        self,
        regime_metrics: Dict[str, Dict],
        save_name: str = "regime_comparison.png"
    ):
        """
        Plot comparison across different market regimes.

        Args:
            regime_metrics: Dictionary mapping regime names to metrics
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Performance Comparison Across Market Regimes',
                    fontsize=16, fontweight='bold')

        regimes = list(regime_metrics.keys())
        colors = {'high_risk': '#E63946', 'high_return': '#06D6A0', 'stable': '#118AB2'}

        metrics_to_plot = [
            ('total_return', 'Total Return (%)', 0),
            ('sharpe_ratio', 'Sharpe Ratio', 1),
            ('max_drawdown', 'Max Drawdown (%)', 2),
            ('win_rate', 'Win Rate', 3),
            ('volatility', 'Volatility (%)', 4),
            ('calmar_ratio', 'Calmar Ratio', 5)
        ]

        for metric_key, metric_name, idx in metrics_to_plot:
            ax = axes[idx // 3, idx % 3]

            values = []
            for regime in regimes:
                val = regime_metrics[regime].get(metric_key, 0)
                if metric_key == 'win_rate':
                    val *= 100  # Convert to percentage
                values.append(val)

            bars = ax.bar(range(len(regimes)), values,
                         color=[colors.get(r, 'gray') for r in regimes])
            ax.set_xticks(range(len(regimes)))
            ax.set_xticklabels([r.replace('_', '\n').title() for r in regimes])
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Regime comparison saved: {save_path}")

    def plot_agent_comparison(
        self,
        agent_returns: Dict[str, np.ndarray],
        save_name: str = "agent_comparison.png"
    ):
        """
        Plot comparison between different agents (HFT, MFT, LFT).

        Args:
            agent_returns: Dictionary mapping agent names to returns
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agent Performance Comparison', fontsize=16, fontweight='bold')

        colors = {'hft': '#E63946', 'mft': '#F77F00', 'lft': '#06D6A0'}

        # 1. Cumulative returns
        ax = axes[0, 0]
        for agent, returns in agent_returns.items():
            cumulative = np.cumprod(1 + returns)
            ax.plot(cumulative, label=agent.upper(), linewidth=2,
                   color=colors.get(agent.lower(), 'gray'))
        ax.set_xlabel('Period')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Returns', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Returns distribution
        ax = axes[0, 1]
        for agent, returns in agent_returns.items():
            ax.hist(returns * 100, bins=30, alpha=0.5, label=agent.upper(),
                   color=colors.get(agent.lower(), 'gray'))
        ax.set_xlabel('Returns (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Risk-Return scatter
        ax = axes[1, 0]
        for agent, returns in agent_returns.items():
            ret = np.mean(returns) * 252 * 100  # Annualized
            vol = np.std(returns) * np.sqrt(252) * 100  # Annualized
            ax.scatter(vol, ret, s=200, label=agent.upper(),
                      color=colors.get(agent.lower(), 'gray'))

        ax.set_xlabel('Volatility (% Annualized)')
        ax.set_ylabel('Return (% Annualized)')
        ax.set_title('Risk-Return Profile', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add diagonal lines for Sharpe ratios
        max_vol = max([np.std(r) * np.sqrt(252) * 100 for r in agent_returns.values()])
        for sharpe in [0.5, 1.0, 1.5, 2.0]:
            x = np.linspace(0, max_vol * 1.2, 100)
            y = sharpe * x
            ax.plot(x, y, '--', alpha=0.3, color='gray', linewidth=1)
            ax.text(max_vol * 1.1, sharpe * max_vol * 1.1, f'S={sharpe}',
                   fontsize=8, alpha=0.5)

        # 4. Metrics comparison table
        ax = axes[1, 1]
        ax.axis('off')

        # Calculate metrics for each agent
        metrics_names = ['Return %', 'Volatility %', 'Sharpe', 'Max DD %']
        table_data = []

        for agent, returns in agent_returns.items():
            ret = np.mean(returns) * 252 * 100
            vol = np.std(returns) * np.sqrt(252) * 100
            sharpe = ret / vol if vol > 0 else 0
            capital = np.cumprod(1 + returns)
            max_dd = np.min((capital - np.maximum.accumulate(capital)) /
                           np.maximum.accumulate(capital)) * 100

            table_data.append([
                agent.upper(),
                f'{ret:.2f}',
                f'{vol:.2f}',
                f'{sharpe:.2f}',
                f'{max_dd:.2f}'
            ])

        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Agent'] + metrics_names,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(metrics_names) + 1):
            table[(0, i)].set_facecolor('#118AB2')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Performance Metrics', fontweight='bold', y=0.95)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Agent comparison saved: {save_path}")

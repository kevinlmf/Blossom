"""
Statistical Validation Framework for RL Trading Strategies

正确的统计验证方法：
1. 多次独立运行（multiple independent runs）
2. 应用中心极限定理（CLT）
3. 样本量计算（power analysis）
4. 统计显著性检验

核心概念：
- 样本 = 一次完整的训练+测试周期
- 需要n≥30次独立运行才能可靠应用CLT
- 使用paired t-test（因为同一数据集上比较）
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class StatisticalTestResult:
    """Statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    is_significant: bool
    interpretation: str


class StatisticalValidator:
    """
    统计验证器 - 确保RL策略显著优于基准策略

    使用方法：
    1. 运行N次独立实验（不同随机种子）
    2. 收集每次运行的性能指标（Sharpe, Return等）
    3. 进行配对t检验
    4. 计算效应量和统计功效
    """

    def __init__(
        self,
        output_dir: str = "outputs/statistical_validation",
        alpha: float = 0.05
    ):
        """
        Initialize statistical validator.

        Args:
            output_dir: Directory to save results
            alpha: Significance level (default: 0.05 for 95% confidence)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha

        print(f"Statistical Validator initialized")
        print(f"Significance level (α): {alpha}")
        print(f"Confidence level: {(1-alpha)*100:.0f}%")

    def validate_superiority(
        self,
        rl_metrics: np.ndarray,
        baseline_metrics: np.ndarray,
        metric_name: str = "Sharpe Ratio",
        alternative: str = "greater"
    ) -> StatisticalTestResult:
        """
        验证RL策略是否显著优于基准策略

        Args:
            rl_metrics: Array of metrics from N independent RL runs
            baseline_metrics: Array of metrics from N independent baseline runs
            metric_name: Name of the metric being tested
            alternative: 'greater' (RL > baseline), 'two-sided', or 'less'

        Returns:
            StatisticalTestResult
        """
        n = len(rl_metrics)

        print(f"\n{'='*80}")
        print(f"STATISTICAL VALIDATION: {metric_name}")
        print(f"{'='*80}")
        print(f"Number of independent runs: {n}")
        print(f"Alternative hypothesis: RL {alternative} baseline")

        # Check if we have enough samples for CLT
        if n < 30:
            print(f"\n⚠️  WARNING: n={n} < 30")
            print(f"   CLT may not hold well. Consider running more experiments.")
            print(f"   Minimum recommended: 30 runs")
            print(f"   Recommended for robust results: 50-100 runs")
        else:
            print(f"\n✅ Sample size adequate (n≥30)")

        # 1. Descriptive statistics
        print(f"\n📊 DESCRIPTIVE STATISTICS")
        print(f"{'─'*80}")
        print(f"{'Metric':<20} {'RL Mean':<15} {'Baseline Mean':<15} {'Difference':<15}")
        print(f"{'─'*80}")
        print(f"{metric_name:<20} {np.mean(rl_metrics):<15.4f} {np.mean(baseline_metrics):<15.4f} "
              f"{np.mean(rl_metrics) - np.mean(baseline_metrics):<15.4f}")
        print(f"{'Std Dev':<20} {np.std(rl_metrics, ddof=1):<15.4f} {np.std(baseline_metrics, ddof=1):<15.4f}")

        # 2. Paired t-test (since same data/conditions)
        # 使用配对t检验因为两个策略在相同的市场数据上测试
        differences = rl_metrics - baseline_metrics

        if alternative == "greater":
            t_stat, p_value = stats.ttest_rel(rl_metrics, baseline_metrics, alternative='greater')
        elif alternative == "less":
            t_stat, p_value = stats.ttest_rel(rl_metrics, baseline_metrics, alternative='less')
        else:
            t_stat, p_value = stats.ttest_rel(rl_metrics, baseline_metrics, alternative='two-sided')

        # 3. Confidence interval for the difference
        ci = self._compute_confidence_interval(differences, self.alpha)

        # 4. Effect size (Cohen's d for paired samples)
        effect_size = self._compute_cohens_d_paired(differences)

        # 5. Statistical power (post-hoc)
        power = self._compute_statistical_power(effect_size, n, self.alpha)

        # 6. Is it significant?
        is_significant = p_value < self.alpha

        # 7. Interpretation
        interpretation = self._interpret_results(
            is_significant, p_value, effect_size, power, alternative
        )

        # Print results
        print(f"\n🧪 STATISTICAL TEST RESULTS")
        print(f"{'─'*80}")
        print(f"Test: Paired t-test")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significant at α={self.alpha}: {'✅ YES' if is_significant else '❌ NO'}")
        print(f"\n95% Confidence Interval for difference: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"Effect Size (Cohen's d): {effect_size:.4f} ({self._interpret_effect_size(effect_size)})")
        print(f"Statistical Power: {power:.4f} ({power*100:.1f}%)")

        print(f"\n💡 INTERPRETATION")
        print(f"{'─'*80}")
        print(interpretation)
        print(f"{'='*80}")

        result = StatisticalTestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            power=power,
            is_significant=is_significant,
            interpretation=interpretation
        )

        # Create visualization
        self._visualize_comparison(rl_metrics, baseline_metrics, metric_name, result)

        return result

    def compute_required_sample_size(
        self,
        expected_effect_size: float,
        desired_power: float = 0.80,
        alpha: float = 0.05
    ) -> int:
        """
        计算需要多少次独立运行才能检测到效应

        Args:
            expected_effect_size: Expected Cohen's d (e.g., 0.5 for medium effect)
            desired_power: Desired statistical power (default: 0.80 = 80%)
            alpha: Significance level

        Returns:
            Required number of independent runs
        """
        print(f"\n{'='*80}")
        print(f"SAMPLE SIZE CALCULATION (Power Analysis)")
        print(f"{'='*80}")
        print(f"Expected effect size (Cohen's d): {expected_effect_size}")
        print(f"Desired statistical power: {desired_power*100:.0f}%")
        print(f"Significance level (α): {alpha}")

        # For paired t-test, approximate formula
        # n ≈ (Z_α + Z_β)² / d²
        # where Z_α = critical value for alpha, Z_β = critical value for power

        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = stats.norm.ppf(desired_power)

        n_required = int(np.ceil(((z_alpha + z_beta) ** 2) / (expected_effect_size ** 2))) + 1

        print(f"\n📊 RESULTS")
        print(f"{'─'*80}")
        print(f"Required number of independent runs: {n_required}")
        print(f"\n💡 RECOMMENDATIONS")
        print(f"{'─'*80}")

        if expected_effect_size >= 0.8:
            print(f"Large effect size detected (d≥0.8)")
            print(f"Minimum runs needed: {n_required}")
            print(f"Recommended: {int(n_required * 1.2)} (with 20% buffer)")
        elif expected_effect_size >= 0.5:
            print(f"Medium effect size expected (0.5≤d<0.8)")
            print(f"Minimum runs needed: {n_required}")
            print(f"Recommended: {int(n_required * 1.3)} (with 30% buffer)")
        else:
            print(f"Small effect size expected (d<0.5)")
            print(f"Minimum runs needed: {n_required}")
            print(f"Recommended: {int(n_required * 1.5)} (with 50% buffer)")
            print(f"\n⚠️  WARNING: Detecting small effects requires many runs!")
            print(f"   Consider if the improvement is practically significant.")

        print(f"\n📝 INTERPRETATION OF EFFECT SIZES")
        print(f"{'─'*80}")
        print(f"Small effect:  d = 0.2 (subtle difference)")
        print(f"Medium effect: d = 0.5 (noticeable difference)")
        print(f"Large effect:  d = 0.8 (obvious difference)")
        print(f"{'='*80}")

        return n_required

    def bootstrap_confidence_interval(
        self,
        rl_metrics: np.ndarray,
        baseline_metrics: np.ndarray,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        使用Bootstrap方法计算置信区间（不依赖CLT假设）

        当样本量较小时（n<30），Bootstrap比t-test更可靠

        Args:
            rl_metrics: RL strategy metrics
            baseline_metrics: Baseline strategy metrics
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default: 0.95)

        Returns:
            (mean_difference, lower_bound, upper_bound)
        """
        print(f"\n{'='*80}")
        print(f"BOOTSTRAP CONFIDENCE INTERVAL")
        print(f"{'='*80}")
        print(f"Bootstrap samples: {n_bootstrap}")
        print(f"Confidence level: {confidence_level*100:.0f}%")

        n = len(rl_metrics)
        differences = rl_metrics - baseline_metrics

        # Bootstrap resampling
        bootstrap_means = []
        rng = np.random.RandomState(42)

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = rng.choice(n, size=n, replace=True)
            bootstrap_sample = differences[indices]
            bootstrap_means.append(np.mean(bootstrap_sample))

        bootstrap_means = np.array(bootstrap_means)

        # Compute confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        mean_diff = np.mean(differences)

        print(f"\n📊 RESULTS")
        print(f"{'─'*80}")
        print(f"Mean difference: {mean_diff:.4f}")
        print(f"{confidence_level*100:.0f}% CI: [{lower:.4f}, {upper:.4f}]")

        if lower > 0:
            print(f"\n✅ The entire confidence interval is positive!")
            print(f"   RL strategy is significantly better with {confidence_level*100:.0f}% confidence.")
        elif upper < 0:
            print(f"\n❌ The entire confidence interval is negative!")
            print(f"   Baseline strategy is significantly better with {confidence_level*100:.0f}% confidence.")
        else:
            print(f"\n⚠️  Confidence interval contains zero.")
            print(f"   Cannot conclude significant difference at {confidence_level*100:.0f}% level.")

        # Plot bootstrap distribution
        self._plot_bootstrap_distribution(bootstrap_means, lower, upper, mean_diff)

        return mean_diff, lower, upper

    def _compute_confidence_interval(
        self,
        differences: np.ndarray,
        alpha: float
    ) -> Tuple[float, float]:
        """Compute confidence interval for mean difference"""
        n = len(differences)
        mean_diff = np.mean(differences)
        se = np.std(differences, ddof=1) / np.sqrt(n)

        # t-critical value
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

        lower = mean_diff - t_crit * se
        upper = mean_diff + t_crit * se

        return (lower, upper)

    def _compute_cohens_d_paired(self, differences: np.ndarray) -> float:
        """
        Compute Cohen's d for paired samples
        d = mean(difference) / std(difference)
        """
        return np.mean(differences) / np.std(differences, ddof=1)

    def _compute_statistical_power(
        self,
        effect_size: float,
        n: int,
        alpha: float
    ) -> float:
        """
        Compute statistical power (post-hoc)
        Power = P(reject H0 | H1 is true)
        """
        # For paired t-test
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n)

        # Critical value
        t_crit = stats.t.ppf(1 - alpha, df=n-1)

        # Power (using non-central t-distribution)
        power = 1 - stats.nct.cdf(t_crit, df=n-1, nc=ncp)

        return power

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_results(
        self,
        is_significant: bool,
        p_value: float,
        effect_size: float,
        power: float,
        alternative: str
    ) -> str:
        """Generate interpretation of results"""
        lines = []

        if is_significant:
            lines.append(f"✅ Statistical Significance: YES (p={p_value:.6f} < α={self.alpha})")
            lines.append(f"   We can reject the null hypothesis.")

            if alternative == "greater":
                lines.append(f"   The RL strategy performs SIGNIFICANTLY BETTER than the baseline.")
            elif alternative == "less":
                lines.append(f"   The baseline performs SIGNIFICANTLY BETTER than the RL strategy.")
            else:
                lines.append(f"   The two strategies perform SIGNIFICANTLY DIFFERENT.")
        else:
            lines.append(f"❌ Statistical Significance: NO (p={p_value:.6f} ≥ α={self.alpha})")
            lines.append(f"   We CANNOT reject the null hypothesis.")
            lines.append(f"   No significant difference detected.")

        lines.append(f"\n📏 Effect Size: {abs(effect_size):.4f} ({self._interpret_effect_size(effect_size)})")

        if abs(effect_size) < 0.2:
            lines.append(f"   The practical difference is negligible.")
        elif abs(effect_size) < 0.5:
            lines.append(f"   The practical difference is small but noticeable.")
        elif abs(effect_size) < 0.8:
            lines.append(f"   The practical difference is moderate and meaningful.")
        else:
            lines.append(f"   The practical difference is large and highly meaningful.")

        lines.append(f"\n⚡ Statistical Power: {power:.4f} ({power*100:.1f}%)")

        if power < 0.70:
            lines.append(f"   ⚠️  LOW POWER - High risk of Type II error (false negative)")
            lines.append(f"   Consider collecting more samples.")
        elif power < 0.80:
            lines.append(f"   ⚠️  MODERATE POWER - Acceptable but could be improved")
        else:
            lines.append(f"   ✅ ADEQUATE POWER - Good ability to detect true effects")

        lines.append(f"\n🎯 Overall Assessment:")

        if is_significant and abs(effect_size) >= 0.5 and power >= 0.80:
            lines.append(f"   🌟 STRONG EVIDENCE: Significant + Meaningful + Well-powered")
            lines.append(f"   The RL strategy demonstrates robust superiority.")
        elif is_significant and abs(effect_size) >= 0.5:
            lines.append(f"   ✅ GOOD EVIDENCE: Significant + Meaningful (but low power)")
            lines.append(f"   Results are promising but more runs recommended.")
        elif is_significant:
            lines.append(f"   ⚠️  WEAK EVIDENCE: Significant but small effect")
            lines.append(f"   The difference may not be practically important.")
        else:
            lines.append(f"   ❌ INSUFFICIENT EVIDENCE: No significant difference")
            lines.append(f"   Either no real difference exists, or sample size is too small.")

        return "\n".join(lines)

    def _visualize_comparison(
        self,
        rl_metrics: np.ndarray,
        baseline_metrics: np.ndarray,
        metric_name: str,
        result: StatisticalTestResult
    ):
        """Create comprehensive visualization"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Box plot comparison
        ax1 = axes[0, 0]
        data_to_plot = [baseline_metrics, rl_metrics]
        bp = ax1.boxplot(data_to_plot, labels=['Baseline', 'RL Strategy'],
                         patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#D62828')
        bp['boxes'][1].set_facecolor('#06A77D')
        ax1.set_ylabel(metric_name, fontsize=12)
        ax1.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add significance indicator
        if result.is_significant:
            y_max = max(np.max(baseline_metrics), np.max(rl_metrics))
            y_min = min(np.min(baseline_metrics), np.min(rl_metrics))
            y_range = y_max - y_min
            h = y_max + 0.1 * y_range
            ax1.plot([1, 1, 2, 2], [h, h+0.05*y_range, h+0.05*y_range, h], 'k-')
            ax1.text(1.5, h+0.07*y_range, f'p={result.p_value:.4f}*',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 2. Violin plot
        ax2 = axes[0, 1]
        positions = [1, 2]
        parts = ax2.violinplot([baseline_metrics, rl_metrics], positions=positions,
                               showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(['Baseline', 'RL Strategy'])
        ax2.set_ylabel(metric_name, fontsize=12)
        ax2.set_title('Density Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Paired differences
        ax3 = axes[1, 0]
        differences = rl_metrics - baseline_metrics
        ax3.hist(differences, bins=20, alpha=0.7, color='#06A77D', edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
        ax3.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean diff = {np.mean(differences):.4f}')

        # Add confidence interval
        ci_lower, ci_upper = result.confidence_interval
        ax3.axvspan(ci_lower, ci_upper, alpha=0.2, color='blue',
                   label=f'95% CI')

        ax3.set_xlabel(f'{metric_name} Difference (RL - Baseline)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Distribution of Differences', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Run-by-run comparison
        ax4 = axes[1, 1]
        runs = np.arange(1, len(rl_metrics) + 1)
        ax4.plot(runs, baseline_metrics, 'o-', color='#D62828', alpha=0.7,
                label='Baseline', linewidth=2, markersize=6)
        ax4.plot(runs, rl_metrics, 's-', color='#06A77D', alpha=0.7,
                label='RL Strategy', linewidth=2, markersize=6)
        ax4.axhline(np.mean(baseline_metrics), color='#D62828', linestyle='--', alpha=0.5)
        ax4.axhline(np.mean(rl_metrics), color='#06A77D', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Run Number', fontsize=12)
        ax4.set_ylabel(metric_name, fontsize=12)
        ax4.set_title('Performance Across Runs', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / f'statistical_comparison_{metric_name.replace(" ", "_").lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved: {save_path}")
        plt.close()

    def _plot_bootstrap_distribution(
        self,
        bootstrap_means: np.ndarray,
        lower: float,
        upper: float,
        mean_diff: float
    ):
        """Plot bootstrap distribution"""

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.hist(bootstrap_means, bins=50, alpha=0.7, color='#06A77D',
               edgecolor='black', density=True)
        ax.axvline(mean_diff, color='blue', linestyle='-', linewidth=2,
                  label=f'Observed mean = {mean_diff:.4f}')
        ax.axvline(lower, color='red', linestyle='--', linewidth=2,
                  label=f'95% CI: [{lower:.4f}, {upper:.4f}]')
        ax.axvline(upper, color='red', linestyle='--', linewidth=2)
        ax.axvline(0, color='black', linestyle=':', linewidth=2,
                  label='No difference (null hypothesis)')

        ax.set_xlabel('Mean Difference', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Bootstrap Distribution of Mean Difference', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / 'bootstrap_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Bootstrap distribution saved: {save_path}")
        plt.close()


def example_usage():
    """
    示例：如何正确使用统计验证
    """
    print("\n" + "="*80)
    print("EXAMPLE: Statistical Validation of RL Trading Strategy")
    print("="*80)

    # 模拟数据：假设运行了50次独立实验
    np.random.seed(42)
    n_runs = 50

    # RL策略：平均Sharpe=1.5, 标准差=0.3
    rl_sharpe_ratios = np.random.normal(1.5, 0.3, n_runs)

    # 基准策略：平均Sharpe=1.2, 标准差=0.25
    baseline_sharpe_ratios = np.random.normal(1.2, 0.25, n_runs)

    # 创建验证器
    validator = StatisticalValidator()

    # 1. 验证显著性
    result = validator.validate_superiority(
        rl_metrics=rl_sharpe_ratios,
        baseline_metrics=baseline_sharpe_ratios,
        metric_name="Sharpe Ratio",
        alternative="greater"
    )

    # 2. Bootstrap置信区间（更稳健）
    mean_diff, lower, upper = validator.bootstrap_confidence_interval(
        rl_metrics=rl_sharpe_ratios,
        baseline_metrics=baseline_sharpe_ratios
    )

    # 3. 计算所需样本量（用于未来实验）
    required_n = validator.compute_required_sample_size(
        expected_effect_size=0.8,  # 预期大效应
        desired_power=0.80,
        alpha=0.05
    )

    print("\n" + "="*80)
    print("EXAMPLE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    example_usage()

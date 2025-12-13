# 📊 Strategy Evaluation Module

全面的交易策略评估系统，提供详细的性能指标、风险分析和可视化报告。

## ✨ 功能特点

### 1. 📈 收益指标
- **总收益率 (Total Return)**: 整个交易期间的总收益
- **年化收益率 (Annualized Return)**: 标准化的年度收益率
- **累积收益 (Cumulative Return)**: 随时间累积的收益
- **平均收益 (Average Return)**: 每期平均收益率

### 2. ⚠️ 风险指标
- **波动率 (Volatility)**: 收益的标准差，衡量价格波动
- **下行波动率 (Downside Volatility)**: 只考虑负收益的波动
- **最大回撤 (Max Drawdown)**: 从峰值到谷底的最大跌幅
- **平均回撤 (Average Drawdown)**: 所有回撤的平均值

### 3. 📊 风险调整后收益指标
- **Sharpe Ratio**: (收益 - 无风险利率) / 波动率
- **Sortino Ratio**: 类似Sharpe，但只考虑下行风险
- **Calmar Ratio**: 年化收益 / 最大回撤
- **Omega Ratio**: 盈利概率 vs 亏损概率

### 4. 🎲 风险度量
- **VaR (Value at Risk)**: 95%和99%置信度的潜在损失
- **CVaR (Conditional VaR)**: 极端情况下的预期损失

### 5. 💰 交易指标
- **胜率 (Win Rate)**: 盈利交易占比
- **盈亏比 (Profit Factor)**: 总盈利 / 总亏损
- **平均盈利/亏损**: 单笔交易的平均盈亏
- **最大连续盈亏**: 最长连胜/连败记录
- **交易次数**: 总交易数量

### 6. ✨ 其他指标
- **Recovery Factor**: 总收益 / 最大回撤
- **Stability**: 累积收益曲线的R²值，越接近1越稳定

## 🚀 快速开始

### 基本使用

```python
from evaluation import StrategyEvaluator
import numpy as np

# 初始化评估器
evaluator = StrategyEvaluator(
    output_dir="outputs/evaluation",
    risk_free_rate=0.02,
    periods_per_year=252  # 每年交易日数
)

# 准备数据（你的策略收益）
returns = np.array([0.01, -0.005, 0.02, ...])  # 期间收益率
capital = np.array([100000, 101000, 100495, ...])  # 资本序列

# 基准数据（买入持有策略）
benchmark_returns = np.array([0.008, -0.003, 0.015, ...])

# 评估策略
metrics = evaluator.evaluate_strategy(
    returns=returns,
    capital_series=capital,
    benchmark_returns=benchmark_returns,
    regime='stable',
    agent_name='my_strategy',
    save_report=True
)

# 查看关键指标
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
print(f"Total Return: {metrics.total_return:.2f}%")
print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
```

### 评估多个Agent

```python
# 准备多个agent的结果
agent_results = {
    'hft': {
        'returns': hft_returns,
        'capital': hft_capital,
        'trades': hft_trades  # 可选
    },
    'mft': {
        'returns': mft_returns,
        'capital': mft_capital
    },
    'lft': {
        'returns': lft_returns,
        'capital': lft_capital
    }
}

# 对比评估
all_metrics = evaluator.evaluate_multiple_agents(
    agent_results=agent_results,
    regime='stable',
    save_report=True
)

# 查看每个agent的表现
for agent, metrics in all_metrics.items():
    print(f"{agent.upper()}: Sharpe={metrics.sharpe_ratio:.3f}")
```

### 评估多个市场周期

```python
# 准备不同周期的结果
regime_results = {
    'high_risk': {
        'returns': crisis_returns,
        'capital': crisis_capital
    },
    'high_return': {
        'returns': bull_returns,
        'capital': bull_capital
    },
    'stable': {
        'returns': stable_returns,
        'capital': stable_capital
    }
}

# 跨周期对比
all_metrics = evaluator.evaluate_multiple_regimes(
    regime_results=regime_results,
    save_report=True
)
```

## 📊 生成的输出

### 1. 可视化报告 (PNG图表)

#### 策略评估报告
包含6个子图的综合报告：
- 累积收益曲线（vs基准）
- 回撤时间序列
- 收益分布直方图
- 滚动Sharpe比率
- 月度收益热图
- 性能指标表格

![Strategy Evaluation](../docs/images/strategy_evaluation_example.png)

#### Agent对比报告
- 各agent累积收益对比
- 收益分布对比
- 风险-收益散点图
- 性能指标对比表

#### 周期对比报告
- 各周期关键指标柱状图对比

### 2. JSON指标文件

```json
{
  "total_return": 25.5,
  "annualized_return": 23.8,
  "sharpe_ratio": 1.25,
  "max_drawdown": -12.3,
  "win_rate": 0.58,
  "volatility": 18.5,
  "var_95": -2.15,
  "cvar_95": -2.85,
  ...
  "timestamp": "2025-11-05T12:25:47",
  "regime": "stable",
  "agent": "my_strategy"
}
```

### 3. 文本摘要报告

```
================================================================================
MULTI-FREQUENCY TRADING SYSTEM - COMPREHENSIVE EVALUATION REPORT
================================================================================

Generated: 2025-11-05 12:25:49

OVERALL SYSTEM PERFORMANCE
--------------------------------------------------------------------------------
  System Sharpe: 1.25
  Total Return: 25.5%
  Max Drawdown: -12.3%

PERFORMANCE BY MARKET REGIME
--------------------------------------------------------------------------------
...
```

## 🎯 在训练系统中使用

评估模块已集成到`train.py`中。训练时会自动：

1. 评估每个周期的策略表现
2. 与基准（买入持有）对比
3. 生成可视化报告
4. 保存详细指标

```bash
# 训练时自动评估
python train.py --mode all_regimes --episodes 500

# 查看生成的评估报告
ls outputs/evaluation/
```

## 📈 输出文件结构

```
outputs/
├── evaluation/
│   ├── strategy_evaluation_stable_system_20251105_122545.png
│   ├── metrics_stable_system_20251105_122547.json
│   ├── agent_comparison_stable.png
│   ├── regime_comparison.png
│   └── summary_report.txt
```

## 🔍 指标解读指南

### Sharpe Ratio (夏普比率)
- **< 0**: 策略表现不如无风险资产
- **0 - 1**: 表现一般
- **1 - 2**: 表现良好
- **> 2**: 优秀表现

### Max Drawdown (最大回撤)
- **< -50%**: 极高风险，难以恢复
- **-30% 到 -50%**: 高风险
- **-10% 到 -30%**: 中等风险
- **> -10%**: 低风险

### Win Rate (胜率)
- **< 40%**: 需要很高的盈亏比才能盈利
- **40% - 50%**: 一般
- **50% - 60%**: 良好
- **> 60%**: 优秀

### Calmar Ratio
- **> 3**: 优秀的风险调整后收益
- **1 - 3**: 良好
- **< 1**: 收益不足以补偿最大回撤风险

## 🧪 测试评估系统

运行测试脚本：

```bash
python test_evaluation.py
```

这将：
1. 生成模拟交易数据
2. 测试单策略评估
3. 测试多agent对比
4. 测试多周期对比
5. 生成所有类型的报告

## 📚 API参考

### StrategyEvaluator

```python
evaluator = StrategyEvaluator(
    output_dir: str = "outputs/evaluation",
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
)
```

### evaluate_strategy()

```python
metrics = evaluator.evaluate_strategy(
    returns: np.ndarray,              # 必需：收益率数组
    capital_series: Optional[np.ndarray] = None,  # 可选：资本序列
    trades: Optional[List[Dict]] = None,          # 可选：交易列表
    benchmark_returns: Optional[np.ndarray] = None,  # 可选：基准收益
    regime: Optional[str] = None,     # 可选：市场周期名称
    agent_name: Optional[str] = None, # 可选：agent名称
    save_report: bool = True          # 是否保存报告
) -> StrategyMetrics
```

### evaluate_multiple_agents()

```python
all_metrics = evaluator.evaluate_multiple_agents(
    agent_results: Dict[str, Dict],   # agent名称 -> 结果字典
    regime: Optional[str] = None,
    save_report: bool = True
) -> Dict[str, StrategyMetrics]
```

### evaluate_multiple_regimes()

```python
all_metrics = evaluator.evaluate_multiple_regimes(
    regime_results: Dict[str, Dict],  # 周期名称 -> 结果字典
    save_report: bool = True
) -> Dict[str, StrategyMetrics]
```

## 💡 最佳实践

### 1. 数据准备
```python
# 确保收益率格式正确（小数形式，不是百分比）
returns = np.array([0.01, -0.005, 0.02])  # ✅ 正确
returns = np.array([1, -0.5, 2])          # ❌ 错误

# 资本序列应该是绝对值
capital = np.array([100000, 101000, 100495])  # ✅ 正确
```

### 2. 基准选择
```python
# 使用买入持有作为基准
benchmark_returns = market_returns  # 市场整体收益

# 或使用无风险利率
risk_free_return = 0.02 / 252  # 日收益率
benchmark_returns = np.full(len(returns), risk_free_return)
```

### 3. 定期评估
```python
# 每训练100个episode评估一次
if episode % 100 == 0:
    metrics = evaluator.evaluate_strategy(
        returns=recent_returns,
        save_report=True
    )
```

### 4. 对比分析
```python
# 同时评估多个版本的策略
strategies = {
    'v1.0': strategy_v1_results,
    'v2.0': strategy_v2_results,
    'v3.0': strategy_v3_results
}

for name, results in strategies.items():
    evaluator.evaluate_strategy(
        returns=results['returns'],
        agent_name=name
    )
```

## ⚙️ 自定义配置

### 调整无风险利率
```python
evaluator = StrategyEvaluator(
    risk_free_rate=0.03  # 3% 年化无风险利率
)
```

### 调整交易频率
```python
# 日频交易
evaluator = StrategyEvaluator(periods_per_year=252)

# 小时频交易
evaluator = StrategyEvaluator(periods_per_year=252*6.5)  # 每天6.5小时

# 分钟频交易
evaluator = StrategyEvaluator(periods_per_year=252*6.5*60)
```

## 🔧 故障排除

### 问题：绘图显示警告

```
RuntimeWarning: divide by zero encountered in divide
```

**解决**：这是正常的，发生在数据不足时。增加数据量即可。

### 问题：指标为NaN或Inf

**原因**：
- 收益率全为0
- 波动率为0
- 数据量太少（< 2个数据点）

**解决**：确保有足够的非零收益数据。

### 问题：可视化图表显示不全

**解决**：增加DPI和图表尺寸：
```python
# 在visualization.py中修改
plt.rcParams['figure.figsize'] = (20, 12)
plt.savefig(path, dpi=300)
```

## 📖 更多示例

查看以下文件获取更多示例：
- `test_evaluation.py` - 完整的测试示例
- `train.py` - 集成到训练流程的示例
- `docs/` - 更多高级用例和文档

## 🤝 贡献

欢迎改进评估模块！可以添加：
- 新的性能指标
- 更多可视化类型
- 导出格式（PDF, Excel等）
- 实时监控功能

---

**Made with ❤️ for Quantitative Trading**

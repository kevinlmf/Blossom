# Blossom · AI-Driven Quantitative Finance Platform

**Multi-Frequency Trading System** powered by Multi-Agent Reinforcement Learning, combining market regime classification, case-based reasoning, and goal-oriented planning to deliver automated strategy generation for quantitative finance.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-green.svg)](https://github.com/google/jax)

## Project Positioning

**AI-Driven Quantitative Finance** - A research platform that combines:
- **Quantitative Finance**: Multi-frequency trading (HFT/MFT/LFT), risk control (CVaR, Max Drawdown), portfolio optimization
- **AI/ML Engineering**: Multi-agent RL (PPO, SAC), deep learning (JAX/Flax), adaptive learning
- **Research Innovation**: Goal-oriented planning, case-based reasoning (CBR), regime-adaptive systems

## Quick Start

```bash
# Interactive menu
bash run.sh

# Auto-detect market regime and train
python train.py --mode auto --episodes 500

# Train with specific goal configuration
python train.py --mode all_regimes \
  --goal-config configs/goals/balanced_growth.yaml \
  --episodes 300
```

**Dependencies**: See `requirements.txt`. Default uses CPU JAX; GPU support available.

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Goal Layer                                                   │
│  Input: U(return, risk, latency, capital, style)             │
│  Output: Goal directive (allocations, risk budgets)          │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Goal Planner + Market Regime Detector                       │
│  Rule-based / HMM-based detection                            │
│  HIGH_RISK | HIGH_RETURN | STABLE                            │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Strategy Memory Bank (CBR)                                   │
│  RETRIEVE → REUSE → REVISE → RETAIN                          │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Stock Selection (Technical + Fundamental)                  │
│  Filters candidate stocks before portfolio construction       │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│  Allocator Agent (Meta-Level PPO)                            │
│  Output: [π_HFT, π_MFT, π_LFT]                               │
└────────────────────┬─────────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
┌─────▼─────┐  ┌─────▼─────┐  ┌────▼──────┐
│ HFT Agent │  │ MFT Agent │  │ LFT Agent │
│   (SAC)   │  │   (SAC)   │  │   (SAC)   │
│  Tick     │  │ Hour/Day  │  │ Portfolio │
└─────┬─────┘  └─────┬─────┘  └────┬──────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
      ┌──────────────▼──────────────┐
      │  Shared Encoder (Transformer)│
      │  Latent Factors: z_t         │
      └──────────────┬──────────────┘
                     │
      ┌──────────────▼──────────────┐
      │   Risk Controller            │
      │   CVaR, Max Drawdown, VaR   │
      └──────────────┬──────────────┘
                     │
      ┌──────────────▼──────────────┐
      │   Hedge Manager (最后一步)   │
      │   Excess → Absolute Return  │
      └──────────────────────────────┘
```

## Key Features

### 1. Stock Selection (选股模块) 🆕
- **Technical Analysis**: Multi-factor technical scoring (momentum, trend, mean reversion, volatility, volume)
- **Fundamental Analysis**: Valuation, profitability, growth, financial health, efficiency metrics
- **Combined Selection**: Integrated technical + fundamental stock screening
- **LFT Integration**: Seamlessly integrated with LFT Agent for portfolio construction

### 2. Multi-Frequency Trading Agents
- **HFT Agent**: Tick-level order execution (order type, quantity, price offset)
- **MFT Agent**: Hourly/daily position management (position size, type)
- **LFT Agent**: Portfolio-level asset allocation (portfolio weights)
- **Action Composition**: `π*(t) = α_H·π_H*(t) + α_M·π_M*(t) + α_L·π_L*(t)`

### 3. Hedging Tools (对冲工具) 🆕
- **Market Neutral Strategy**: Convert excess returns to absolute returns
- **Dynamic Delta Hedging**: Adaptive hedging based on market volatility
- **Beta Hedging**: Beta-based market risk neutralization
- **Multiple Strategies**: Pair trading, index hedging support

### 4. Market Regime Detection 🆕
- **Rule-Based Method**: Fast, interpretable threshold-based classification
- **HMM Method**: Hidden Markov Model with state transition probabilities and prediction capability
- **Regime Types**: High Risk, High Return, Stable periods
- **Adaptive Parameters**: Dynamic parameter adjustment based on detected regime

### 5. Case-Based Reasoning (CBR)
- Strategy memory bank for warm-starting agents
- Similarity-based retrieval of optimal strategies
- Continuous learning and strategy evolution

### 6. Goal-Oriented Planning
- YAML/JSON goal configuration
- Personalized utility functions
- Goal-conditioned reward shaping
- Goal satisfaction tracking and auditing

### 7. Risk Control
- Real-time CVaR and Max Drawdown monitoring
- Dynamic position adjustment based on risk levels
- Goal-constrained risk management

### 8. EM-Based Latent Variable Learning
- Expectation-Maximization algorithm for learning latent factors
- Shared encoder extracts factors explaining asset returns
- R-squared optimization for return prediction

## Execution Flow

**Correct execution order**:
```
1. Stock Selection (筛选股票池)
   ↓
2. Agents Work Independently (HFT/MFT/LFT生成动作)
   ↓
3. Allocator (资本分配)
   ↓
4. Risk Control (风险调整)
   ↓
5. Hedge (对冲 - 最后一步，转换为绝对收益)
```

## Project Structure

```
Blossom/
├── agent/              # Multi-frequency agents (HFT/MFT/LFT)
│   ├── allocator/      # Capital allocation agent
│   ├── hft_agent/      # High-frequency trading agent (SAC)
│   ├── mft_agent/      # Medium-frequency trading agent (SAC)
│   ├── lft_agent/      # Low-frequency trading agent (SAC)
│   │   └── selection/  # Stock selection for LFT 🆕
│   ├── memory/         # Strategy memory bank (CBR)
│   └── planner/        # Goal planner
├── stock_selection/    # Stock selection module 🆕
├── hedging/            # Hedging tools 🆕
├── experiments/        # Experiment runners and analysis
│   ├── market_regime_detector.py  # Rule-based detection
│   └── hmm_regime_detector.py     # HMM-based detection 🆕
├── shared_encoder/     # Shared encoder (Transformer/LSTM/EM)
├── risk_controller/    # Risk control (CVaR, Max Drawdown)
├── goal/               # Goal-oriented planning
├── evaluation/         # Performance evaluation and validation
└── docs/               # Documentation
```

## Market Regime Detection

### Rule-Based Method (Current)
- **Method**: Threshold-based classification using volatility, returns, drawdown
- **Pros**: Fast, interpretable, no training needed
- **Use Case**: Real-time detection, quick analysis

### HMM Method (New) 🆕
- **Method**: Hidden Markov Model with state transition probabilities
- **Pros**: More accurate, can predict future regimes, learns from data
- **Use Case**: Precise detection, regime prediction

```python
from experiments import HMMRegimeDetector

hmm_detector = HMMRegimeDetector(n_states=3, learn_params=True)
hmm_detector.fit(prices, dates)
periods = hmm_detector.detect_specific_periods(prices, dates)

# Predict future regime
future_probs = hmm_detector.predict_next_regime(prices, n_steps=5)
```

## Usage Examples

### Stock Selection
```python
from stock_selection import StockSelector

selector = StockSelector(technical_weight=0.5, fundamental_weight=0.5, top_k=10)
selected = selector.select_stocks(stock_data, symbols)
```

### Hedging
```python
from hedging import HedgeManager, HedgeStrategy

hedge_manager = HedgeManager(strategy=HedgeStrategy.MARKET_NEUTRAL)
result = hedge_manager.hedge_portfolio(
    excess_return=0.05, portfolio_value=100000, benchmark_return=0.10
)
absolute_return = result.absolute_return
```

### LFT with Stock Selection
```python
from agent.lft_agent import LFTStockSelector

stock_selector = LFTStockSelector(top_k=10, rebalance_frequency=20)
selected_symbols, scores = stock_selector.select_stocks_for_lft(
    stock_data, symbols, current_step=0
)
```

## Evaluation & Validation

- **Performance Metrics**: Sharpe ratio, total return, max drawdown, win rate, CVaR
- **Statistical Validation**: Multiple independent runs with significance testing
- **Goal Audit**: Goal satisfaction tracking and reporting
- **Regime Analysis**: Performance breakdown by market regime

## Experimental Results (30 Independent Runs)

Based on 30 independent experimental runs across three distinct market regimes:

**High Return Regime (Bull Market)**:
- Sharpe Ratio: **4.76 ± 0.67** | Total Return: **177.90% ± 41.25%** | Max DD: **-5.74% ± 1.17%** | Win Rate: **61.70%**
- Encoder: Ensemble (Transformer + LSTM) | Consistent outperformance with effective risk management

**Stable Regime (Normal Market)**:
- Sharpe Ratio: **4.46 ± 1.01** | Total Return: **58.87% ± 15.66%** | Max DD: **-4.08% ± 0.94%** | Win Rate: **61.34%**
- Encoder: Transformer | Stable returns with excellent risk-adjusted performance

**High Risk Regime (Crisis Period)**:
- Sharpe Ratio: **-0.95 ± 2.09** | Total Return: **-10.43% ± 23.30%** | Max DD: **-28.50% ± 11.24%** | Win Rate: **47.70%**
- Encoder: LSTM | Defensive behavior during extreme market conditions

**Key Findings**:
1. **Regime Adaptability**: System successfully adapts encoder architecture (LSTM/Transformer/Ensemble) based on market regime
2. **Consistent Performance**: Low variance in Sharpe ratios for High Return and Stable regimes
3. **Risk Management**: Controlled drawdowns demonstrate effective CVaR and Max Drawdown monitoring
4. **CBR Benefits**: Warm-starting significantly improves training efficiency and final performance
5. **Multi-Agent Coordination**: Hierarchical architecture successfully coordinates across time horizons

*Results based on 30 independent runs (500 episodes each) using predefined market periods (COVID-19 crash, post-COVID rally, pre-COVID stable market).*

## Performance Optimizations

Memory Bank Caching | Incremental Updates | Parallel Training | JAX JIT Compilation

## Documentation

See `docs/` directory for detailed documentation on workflow, market regime detection, HMM methods, EM training, and evaluation metrics.

## Research & Innovation

- **Multi-Agent RL**: Hierarchical agent architecture with meta-level coordination
- **Goal-Conditioned RL**: Goal-oriented reward shaping and planning
- **Case-Based Reasoning**: Strategy memory bank for warm-starting and continuous learning
- **Regime-Adaptive Systems**: Automatic market regime detection (Rule-based + HMM)
- **Stock Selection**: Technical + fundamental multi-factor screening
- **Hedging**: Market-neutral strategies for absolute returns
- **Latent Variable Learning**: EM algorithm for discovering factors explaining returns

## Installation

```bash
# Clone repository
git clone https://github.com/kevinlmf/Blossom
cd Blossom

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```





---
Name it “Blossom,” hoping our life can bloom like flowers, and that we can walk in nature to find their beauty.

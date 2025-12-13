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

## Installation

```bash
# Clone the repository
git clone https://github.com/kevinlmf/Blossom
cd Blossom

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---
### Interactive Menu
Launch the interactive interface:
```bash
bash run.sh
```

### Auto-Detect Market Regime and Train
Automatically detect the market regime and train agents:
```bash
python train.py --mode auto --episodes 500
```

### Train with a Specific Goal Configuration
Train across all regimes with a predefined goal setup:
```bash
python train.py --mode all_regimes \
  --goal-config configs/goals/balanced_growth.yaml \
  --episodes 300
```



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

### 1. Stock Selection
- **Technical Analysis**: Multi-factor technical scoring (momentum, trend, mean reversion, volatility, volume)
- **Fundamental Analysis**: Valuation, profitability, growth, financial health, efficiency metrics
- **Combined Selection**: Integrated technical + fundamental stock screening
- **LFT Integration**: Seamlessly integrated with LFT Agent for portfolio construction

### 2. Multi-Frequency Trading Agents
- **HFT Agent**: Tick-level order execution (order type, quantity, price offset)
- **MFT Agent**: Hourly/daily position management (position size, type)
- **LFT Agent**: Portfolio-level asset allocation (portfolio weights)
- **Action Composition**: `π*(t) = α_H·π_H*(t) + α_M·π_M*(t) + α_L·π_L*(t)`

### 3. Hedging Tools
- **Market Neutral Strategy**: Convert excess returns to absolute returns
- **Dynamic Delta Hedging**: Adaptive hedging based on market volatility
- **Beta Hedging**: Beta-based market risk neutralization
- **Multiple Strategies**: Pair trading, index hedging support

### 4. Market Regime Detection
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
1. Stock Selection
   ↓
2. Agents Work Independently
   ↓
3. Allocator
   ↓
4. Risk Control
   ↓
5. Hedge
```

## Project Structure

```
Blossom/
├── agent/              # Multi-frequency agents (HFT/MFT/LFT)
│   ├── allocator/      # Capital allocation agent
│   ├── hft_agent/      # High-frequency trading agent (SAC)
│   ├── mft_agent/      # Medium-frequency trading agent (SAC)
│   ├── lft_agent/      # Low-frequency trading agent (SAC)
│   │   └── selection/  # Stock selection for LFT 
│   ├── memory/         # Strategy memory bank (CBR)
│   └── planner/        # Goal planner
├── stock_selection/    # Stock selection module 
├── hedging/            # Hedging tools 
├── experiments/        # Experiment runners and analysis
│   ├── market_regime_detector.py  # Rule-based detection
│   └── hmm_regime_detector.py     # HMM-based detection 
├── shared_encoder/     # Shared encoder (Transformer/LSTM/EM)
├── risk_controller/    # Risk control (CVaR, Max Drawdown)
├── goal/               # Goal-oriented planning
├── evaluation/         # Performance evaluation and validation
└── docs/               # Documentation
```

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


## Research & Innovation

- **Multi-Agent RL**: Hierarchical agent architecture with meta-level coordination
- **Goal-Conditioned RL**: Goal-oriented reward shaping and planning
- **Case-Based Reasoning**: Strategy memory bank for warm-starting and continuous learning
- **Regime-Adaptive Systems**: Automatic market regime detection (Rule-based + HMM)
- **Stock Selection**: Technical + fundamental multi-factor screening
- **Hedging**: Market-neutral strategies for absolute returns
- **Latent Variable Learning**: EM algorithm for discovering factors explaining returns

---
## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Disclaimer

⚠️ **Important**: This project is provided **for educational, academic research, and learning purposes only**. Commercial use is strictly prohibited. Please read the [DISCLAIMER.md](DISCLAIMER.md) file carefully before using this project.

---

Name it “Blossom,” hoping our life can bloom like flowers, also we can walk in nature to find beautiful flowers.

"""
HMM-based Market Regime Detector

使用隐马尔可夫模型（Hidden Markov Model）进行市场周期检测。

相比基于规则的方法，HMM的优势：
1. 考虑状态转换概率
2. 给出概率输出而非硬分类
3. 可以从数据中学习参数
4. 有预测能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    print("Warning: hmmlearn not installed. Install with: pip install hmmlearn")


class MarketRegime(Enum):
    """Market regime types."""
    HIGH_RISK = "high_risk"
    HIGH_RETURN = "high_return"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class HMMRegimePeriod:
    """HMM检测到的市场周期"""
    regime: MarketRegime
    start_date: datetime
    end_date: datetime
    start_idx: int
    end_idx: int
    probability: float  # HMM给出的概率
    volatility: float
    avg_return: float


class HMMRegimeDetector:
    """
    基于HMM的市场周期检测器
    
    使用隐马尔可夫模型检测市场周期，相比规则方法：
    - 考虑状态转换概率
    - 给出概率输出
    - 可以从数据学习参数
    """

    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 2,  # 收益率和波动率
        covariance_type: str = "full",  # "full", "diag", "spherical", "tied"
        n_iter: int = 100,
        random_state: int = 42,
        learn_params: bool = True  # 是否从数据学习参数
    ):
        """
        Args:
            n_states: 隐藏状态数量（3个周期）
            n_features: 观测特征数量（收益率、波动率等）
            covariance_type: 协方差矩阵类型
            n_iter: EM算法迭代次数
            random_state: 随机种子
            learn_params: 是否从数据学习参数（True）或使用预设参数（False）
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")
        
        self.n_states = n_states
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.learn_params = learn_params
        
        # 初始化HMM模型
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        
        # 状态到周期的映射（需要根据学习结果调整）
        self.state_to_regime = {
            0: MarketRegime.HIGH_RISK,
            1: MarketRegime.HIGH_RETURN,
            2: MarketRegime.STABLE
        }
        
        self.is_fitted = False

    def prepare_observations(
        self,
        prices: np.ndarray,
        volatility_window: int = 20
    ) -> np.ndarray:
        """
        准备HMM的观测序列
        
        Args:
            prices: 价格序列
            volatility_window: 波动率计算窗口
        
        Returns:
            观测矩阵 [T, n_features]
        """
        # 计算收益率
        returns = np.diff(prices) / prices[:-1]
        
        # 计算滚动波动率
        volatility = self._rolling_std(returns, volatility_window)
        
        # 对齐长度（volatility比returns少一个）
        returns = returns[volatility_window-1:]
        
        # 组合观测特征
        observations = np.column_stack([
            returns,
            volatility[volatility_window-1:]
        ])
        
        # 移除NaN
        valid_mask = ~np.isnan(observations).any(axis=1)
        observations = observations[valid_mask]
        
        return observations

    def fit(
        self,
        prices: np.ndarray,
        dates: Optional[List[datetime]] = None
    ):
        """
        训练HMM模型（学习参数）
        
        Args:
            prices: 价格序列
            dates: 日期序列（可选）
        """
        observations = self.prepare_observations(prices)
        
        if len(observations) < self.n_states * 2:
            raise ValueError(f"Not enough data. Need at least {self.n_states * 2} observations.")
        
        # 训练HMM模型
        self.model.fit(observations)
        self.is_fitted = True
        
        # 根据学习结果调整状态映射
        self._adjust_state_mapping(observations)
        
        print(f"✅ HMM模型训练完成")
        print(f"   状态数量: {self.n_states}")
        print(f"   观测特征数: {self.n_features}")
        print(f"   训练样本数: {len(observations)}")

    def detect_regimes(
        self,
        prices: np.ndarray,
        dates: Optional[List[datetime]] = None
    ) -> List[HMMRegimePeriod]:
        """
        检测市场周期
        
        Args:
            prices: 价格序列
            dates: 日期序列（可选）
        
        Returns:
            检测到的周期列表
        """
        if not self.is_fitted:
            # 如果没有训练，先训练
            self.fit(prices, dates)
        
        observations = self.prepare_observations(prices)
        
        # 使用Viterbi算法解码最可能的状态序列
        states = self.model.predict(observations)
        
        # 计算每个状态的概率
        state_probs = self.model.predict_proba(observations)
        
        # 将状态序列转换为周期
        periods = self._states_to_periods(
            states, state_probs, observations, dates
        )
        
        return periods

    def detect_specific_periods(
        self,
        prices: np.ndarray,
        dates: Optional[List[datetime]] = None,
        min_period_length: int = 20
    ) -> Dict[str, List[HMMRegimePeriod]]:
        """
        检测并分类特定周期
        
        Args:
            prices: 价格序列
            dates: 日期序列（可选）
            min_period_length: 最小周期长度
        
        Returns:
            按周期类型分类的周期字典
        """
        all_periods = self.detect_regimes(prices, dates)
        
        categorized = {
            'high_risk': [],
            'high_return': [],
            'stable': []
        }
        
        for period in all_periods:
            period_length = period.end_idx - period.start_idx
            
            if period_length < min_period_length:
                continue
            
            if period.regime == MarketRegime.HIGH_RISK:
                categorized['high_risk'].append(period)
            elif period.regime == MarketRegime.HIGH_RETURN:
                categorized['high_return'].append(period)
            elif period.regime == MarketRegime.STABLE:
                categorized['stable'].append(period)
        
        return categorized

    def predict_next_regime(
        self,
        prices: np.ndarray,
        n_steps: int = 1
    ) -> Dict[str, float]:
        """
        预测未来n步的市场周期概率
        
        Args:
            prices: 当前价格序列
            n_steps: 预测步数
        
        Returns:
            未来周期的概率分布
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        observations = self.prepare_observations(prices)
        
        # 获取当前状态
        current_state = self.model.predict(observations[-1:])[0]
        
        # 获取状态转换矩阵
        transition_matrix = self.model.transmat_
        
        # 计算n步后的状态概率
        state_probs = transition_matrix[current_state]
        for _ in range(n_steps - 1):
            state_probs = state_probs @ transition_matrix
        
        # 转换为周期概率
        regime_probs = {}
        for state_idx, prob in enumerate(state_probs):
            regime = self.state_to_regime.get(state_idx, MarketRegime.UNKNOWN)
            regime_name = regime.value
            regime_probs[regime_name] = float(prob)
        
        return regime_probs

    def get_regime_probabilities(
        self,
        prices: np.ndarray
    ) -> np.ndarray:
        """
        获取每个时间点的周期概率分布
        
        Args:
            prices: 价格序列
        
        Returns:
            概率矩阵 [T, n_states]
        """
        if not self.is_fitted:
            self.fit(prices)
        
        observations = self.prepare_observations(prices)
        state_probs = self.model.predict_proba(observations)
        
        return state_probs

    def _adjust_state_mapping(
        self,
        observations: np.ndarray
    ):
        """
        根据学习结果调整状态到周期的映射
        
        策略：根据每个状态的观测特征（收益率、波动率）来判断周期类型
        """
        # 获取每个状态的均值
        means = self.model.means_  # [n_states, n_features]
        
        # 根据收益率和波动率特征排序
        # 假设：收益率高 -> HIGH_RETURN, 波动率高 -> HIGH_RISK, 其他 -> STABLE
        state_scores = []
        for i, mean in enumerate(means):
            return_mean = mean[0]
            vol_mean = mean[1]
            
            # 评分：收益率越高越好，波动率越低越好
            score = return_mean - vol_mean
            state_scores.append((i, score, return_mean, vol_mean))
        
        # 排序：score从高到低
        state_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 映射：最高分 -> HIGH_RETURN, 最低分 -> HIGH_RISK, 中间 -> STABLE
        if len(state_scores) >= 3:
            self.state_to_regime[state_scores[0][0]] = MarketRegime.HIGH_RETURN
            self.state_to_regime[state_scores[1][0]] = MarketRegime.STABLE
            self.state_to_regime[state_scores[2][0]] = MarketRegime.HIGH_RISK
        elif len(state_scores) == 2:
            self.state_to_regime[state_scores[0][0]] = MarketRegime.HIGH_RETURN
            self.state_to_regime[state_scores[1][0]] = MarketRegime.HIGH_RISK

    def _states_to_periods(
        self,
        states: np.ndarray,
        state_probs: np.ndarray,
        observations: np.ndarray,
        dates: Optional[List[datetime]] = None
    ) -> List[HMMRegimePeriod]:
        """将状态序列转换为周期列表"""
        if len(states) == 0:
            return []
        
        periods = []
        current_state = states[0]
        start_idx = 0
        
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != current_state:
                # 计算周期统计
                period_obs = observations[start_idx:i]
                period_probs = state_probs[start_idx:i]
                
                avg_prob = np.mean(period_probs[:, current_state])
                avg_return = np.mean(period_obs[:, 0])
                avg_vol = np.mean(period_obs[:, 1])
                
                # 获取周期类型
                regime = self.state_to_regime.get(int(current_state), MarketRegime.UNKNOWN)
                
                # 日期
                start_date = dates[start_idx] if dates and start_idx < len(dates) else None
                end_date = dates[i-1] if dates and i-1 < len(dates) else None
                
                period = HMMRegimePeriod(
                    regime=regime,
                    start_date=start_date,
                    end_date=end_date,
                    start_idx=start_idx,
                    end_idx=i,
                    probability=float(avg_prob),
                    volatility=float(avg_vol),
                    avg_return=float(avg_return)
                )
                
                periods.append(period)
                
                if i < len(states):
                    current_state = states[i]
                    start_idx = i
        
        return periods

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """计算滚动标准差"""
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1])
        return result

    def print_model_info(self):
        """打印HMM模型信息"""
        if not self.is_fitted:
            print("Model not fitted yet.")
            return
        
        print("\n" + "=" * 80)
        print("HMM MODEL INFORMATION")
        print("=" * 80)
        
        print("\n状态转换矩阵 (Transition Matrix):")
        print(self.model.transmat_)
        
        print("\n初始状态概率 (Initial Probabilities):")
        print(self.model.startprob_)
        
        print("\n状态均值 (State Means):")
        print(self.model.means_)
        
        print("\n状态协方差 (State Covariances):")
        print(self.model.covars_)
        
        print("\n状态到周期映射:")
        for state_idx, regime in self.state_to_regime.items():
            print(f"  State {state_idx} -> {regime.value}")
        
        print("=" * 80)


class POMPRegimeDetector:
    """
    POMP (Partially Observed Markov Process) 市场周期检测器
    
    POMP相比HMM的优势：
    - 连续状态空间
    - 更灵活的状态演化模型
    - 可以建模更复杂的动态过程
    
    注意：POMP实现较复杂，这里提供基础框架
    """
    
    def __init__(self):
        """
        POMP实现需要：
        1. 状态演化模型（SDE）
        2. 观测模型
        3. 粒子滤波或卡尔曼滤波进行状态估计
        """
        # TODO: 实现POMP版本
        raise NotImplementedError("POMP implementation is more complex. Consider using HMM first.")


def compare_methods(
    prices: np.ndarray,
    dates: Optional[List[datetime]] = None
):
    """
    对比规则方法和HMM方法
    
    Args:
        prices: 价格序列
        dates: 日期序列
    """
    from experiments.market_regime_detector import MarketRegimeDetector as RuleBasedDetector
    
    print("\n" + "=" * 80)
    print("市场周期检测方法对比")
    print("=" * 80)
    
    # 规则方法
    print("\n1. 规则方法（当前方法）:")
    rule_detector = RuleBasedDetector()
    rule_periods = rule_detector.detect_specific_periods(prices, dates)
    
    for regime_name, periods in rule_periods.items():
        print(f"  {regime_name}: {len(periods)} periods")
    
    # HMM方法
    if HMMLEARN_AVAILABLE:
        print("\n2. HMM方法:")
        hmm_detector = HMMRegimeDetector()
        hmm_periods = hmm_detector.detect_specific_periods(prices, dates)
        
        for regime_name, periods in hmm_periods.items():
            print(f"  {regime_name}: {len(periods)} periods")
            if periods:
                avg_prob = np.mean([p.probability for p in periods])
                print(f"    平均概率: {avg_prob:.3f}")
        
        # 打印HMM模型信息
        hmm_detector.print_model_info()
    else:
        print("\n2. HMM方法: 需要安装hmmlearn")
        print("   安装命令: pip install hmmlearn")
    
    print("\n" + "=" * 80)




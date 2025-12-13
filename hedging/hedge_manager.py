"""
Hedge Manager

对冲管理模块，将超额收益转换为绝对收益。
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import chex


class HedgeStrategy(Enum):
    """对冲策略类型"""
    MARKET_NEUTRAL = "market_neutral"  # 市场中性
    PAIR_TRADING = "pair_trading"  # 配对交易
    INDEX_HEDGE = "index_hedge"  # 指数对冲
    DYNAMIC_DELTA = "dynamic_delta"  # 动态Delta对冲


@dataclass
class HedgeResult:
    """对冲结果"""
    hedge_ratio: float  # 对冲比例 [0, 1]
    hedge_position: float  # 对冲仓位
    hedged_return: float  # 对冲后收益
    excess_return: float  # 原始超额收益
    absolute_return: float  # 绝对收益
    hedge_cost: float  # 对冲成本
    strategy: HedgeStrategy


class HedgeManager:
    """
    对冲管理器
    
    功能：
    1. 计算最优对冲比例
    2. 执行对冲策略
    3. 将超额收益转换为绝对收益
    """

    def __init__(
        self,
        strategy: HedgeStrategy = HedgeStrategy.MARKET_NEUTRAL,
        hedge_cost: float = 0.0001,  # 对冲成本（如期货交易成本）
        min_hedge_ratio: float = 0.0,
        max_hedge_ratio: float = 1.0,
        lookback_window: int = 60
    ):
        """
        Args:
            strategy: 对冲策略
            hedge_cost: 对冲成本（每次对冲的成本率）
            min_hedge_ratio: 最小对冲比例
            max_hedge_ratio: 最大对冲比例
            lookback_window: 计算对冲比例的历史窗口
        """
        self.strategy = strategy
        self.hedge_cost = hedge_cost
        self.min_hedge_ratio = min_hedge_ratio
        self.max_hedge_ratio = max_hedge_ratio
        self.lookback_window = lookback_window
        
        # 历史数据缓存
        self.portfolio_returns_history = []
        self.benchmark_returns_history = []

    def compute_hedge_ratio(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        method: str = "beta"
    ) -> float:
        """
        计算最优对冲比例
        
        Args:
            portfolio_returns: 组合收益序列
            benchmark_returns: 基准收益序列（如市场指数）
            method: 计算方法 ('beta', 'correlation', 'min_variance')
        
        Returns:
            对冲比例 [0, 1]
        """
        if len(portfolio_returns) < 10 or len(benchmark_returns) < 10:
            return 0.0
        
        # 确保长度一致
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
        
        if method == "beta":
            # Beta对冲：hedge_ratio = beta
            beta = self._compute_beta(portfolio_returns, benchmark_returns)
            hedge_ratio = np.clip(beta, self.min_hedge_ratio, self.max_hedge_ratio)
        
        elif method == "correlation":
            # 相关性对冲：hedge_ratio = correlation
            correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
            hedge_ratio = np.clip(abs(correlation), self.min_hedge_ratio, self.max_hedge_ratio)
        
        elif method == "min_variance":
            # 最小方差对冲
            hedge_ratio = self._compute_min_variance_hedge(portfolio_returns, benchmark_returns)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(hedge_ratio)

    def hedge_portfolio(
        self,
        excess_return: float,
        portfolio_value: float,
        benchmark_return: float,
        current_hedge_ratio: Optional[float] = None
    ) -> HedgeResult:
        """
        对组合进行对冲
        
        Args:
            excess_return: 超额收益（相对于基准）
            portfolio_value: 当前组合价值
            benchmark_return: 基准收益
            current_hedge_ratio: 当前对冲比例（如果None，会重新计算）
        
        Returns:
            HedgeResult对象
        """
        # 更新历史数据
        self.portfolio_returns_history.append(excess_return + benchmark_return)
        self.benchmark_returns_history.append(benchmark_return)
        
        if len(self.portfolio_returns_history) > self.lookback_window:
            self.portfolio_returns_history.pop(0)
            self.benchmark_returns_history.pop(0)
        
        # 计算对冲比例
        if current_hedge_ratio is None:
            if len(self.portfolio_returns_history) >= 10:
                portfolio_returns = np.array(self.portfolio_returns_history)
                benchmark_returns = np.array(self.benchmark_returns_history)
                hedge_ratio = self.compute_hedge_ratio(portfolio_returns, benchmark_returns)
            else:
                hedge_ratio = 0.0
        else:
            hedge_ratio = current_hedge_ratio
        
        # 根据策略调整对冲比例
        hedge_ratio = self._adjust_hedge_ratio_by_strategy(
            hedge_ratio,
            excess_return,
            benchmark_return
        )
        
        # 计算对冲仓位
        hedge_position = portfolio_value * hedge_ratio
        
        # 计算对冲后收益
        # 绝对收益 = 超额收益 - hedge_ratio * 基准收益 - 对冲成本
        hedged_return = excess_return - hedge_ratio * benchmark_return
        absolute_return = hedged_return - self.hedge_cost * abs(hedge_ratio)
        
        # 计算对冲成本
        hedge_cost = self.hedge_cost * abs(hedge_ratio) * portfolio_value
        
        return HedgeResult(
            hedge_ratio=float(hedge_ratio),
            hedge_position=float(hedge_position),
            hedged_return=float(hedged_return),
            excess_return=float(excess_return),
            absolute_return=float(absolute_return),
            hedge_cost=float(hedge_cost),
            strategy=self.strategy
        )

    def _adjust_hedge_ratio_by_strategy(
        self,
        base_hedge_ratio: float,
        excess_return: float,
        benchmark_return: float
    ) -> float:
        """根据策略调整对冲比例"""
        if self.strategy == HedgeStrategy.MARKET_NEUTRAL:
            # 市场中性：完全对冲市场风险
            return np.clip(base_hedge_ratio, self.min_hedge_ratio, self.max_hedge_ratio)
        
        elif self.strategy == HedgeStrategy.DYNAMIC_DELTA:
            # 动态Delta对冲：根据市场波动调整
            volatility = abs(benchmark_return)
            if volatility > 0.02:  # 高波动时增加对冲
                adjusted_ratio = min(base_hedge_ratio * 1.2, self.max_hedge_ratio)
            else:
                adjusted_ratio = base_hedge_ratio
            return np.clip(adjusted_ratio, self.min_hedge_ratio, self.max_hedge_ratio)
        
        elif self.strategy == HedgeStrategy.PAIR_TRADING:
            # 配对交易：当超额收益过大时减少对冲
            if abs(excess_return) > 0.05:
                adjusted_ratio = base_hedge_ratio * 0.8
            else:
                adjusted_ratio = base_hedge_ratio
            return np.clip(adjusted_ratio, self.min_hedge_ratio, self.max_hedge_ratio)
        
        else:
            return np.clip(base_hedge_ratio, self.min_hedge_ratio, self.max_hedge_ratio)

    def _compute_beta(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """计算Beta"""
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance < 1e-8:
            return 0.0
        
        beta = covariance / benchmark_variance
        return float(beta)

    def _compute_min_variance_hedge(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """计算最小方差对冲比例"""
        if len(portfolio_returns) < 2:
            return 0.0
        
        # 最小方差对冲：h = Cov(P, B) / Var(B)
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance < 1e-8:
            return 0.0
        
        hedge_ratio = covariance / benchmark_variance
        return float(np.clip(hedge_ratio, self.min_hedge_ratio, self.max_hedge_ratio))

    def convert_to_absolute_returns(
        self,
        excess_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        portfolio_values: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[HedgeResult]]:
        """
        将超额收益序列转换为绝对收益序列
        
        Args:
            excess_returns: 超额收益序列
            benchmark_returns: 基准收益序列
            portfolio_values: 组合价值序列（可选，用于计算对冲成本）
        
        Returns:
            (绝对收益序列, 对冲结果列表)
        """
        if portfolio_values is None:
            portfolio_values = np.ones_like(excess_returns) * 100000  # 默认100k
        
        absolute_returns = []
        hedge_results = []
        
        for i in range(len(excess_returns)):
            excess_ret = excess_returns[i]
            benchmark_ret = benchmark_returns[i]
            portfolio_value = portfolio_values[i]
            
            result = self.hedge_portfolio(
                excess_return=excess_ret,
                portfolio_value=portfolio_value,
                benchmark_return=benchmark_ret
            )
            
            absolute_returns.append(result.absolute_return)
            hedge_results.append(result)
        
        return np.array(absolute_returns), hedge_results

    def reset(self):
        """重置历史数据"""
        self.portfolio_returns_history = []
        self.benchmark_returns_history = []

    def get_statistics(self) -> Dict[str, float]:
        """获取对冲统计信息"""
        if len(self.portfolio_returns_history) == 0:
            return {
                'avg_hedge_ratio': 0.0,
                'total_hedge_cost': 0.0,
                'num_hedges': 0
            }
        
        portfolio_returns = np.array(self.portfolio_returns_history)
        benchmark_returns = np.array(self.benchmark_returns_history)
        
        avg_hedge_ratio = self.compute_hedge_ratio(portfolio_returns, benchmark_returns)
        
        return {
            'avg_hedge_ratio': float(avg_hedge_ratio),
            'total_hedge_cost': float(self.hedge_cost * len(self.portfolio_returns_history)),
            'num_hedges': len(self.portfolio_returns_history)
        }





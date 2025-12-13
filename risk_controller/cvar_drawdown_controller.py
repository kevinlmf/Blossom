"""
CVaR + Max Drawdown Risk Controller

专注于实际风控效果，结合CVaR和Max Drawdown进行实时风险管理。
不需要复杂的统计验证，而是关注实际的风控效果。

核心思想：
1. CVaR监控：实时监控组合的尾部风险
2. Max Drawdown监控：防止组合价值大幅回撤
3. 联合风控：当任一指标超过阈值时触发风控措施
4. 动态调整：根据市场状态动态调整风控阈值
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class RiskLimits:
    """风控限制"""
    max_cvar_95: float = -0.05  # 最大CVaR (95%置信度)
    max_drawdown: float = -0.15  # 最大回撤限制 (-15%)
    max_daily_loss: float = -0.03  # 最大单日损失 (-3%)
    warning_cvar: float = -0.03  # CVaR警告阈值
    warning_drawdown: float = -0.10  # 回撤警告阈值


@dataclass
class RiskStatus:
    """风控状态"""
    is_safe: bool = True
    cvar_95: float = 0.0
    current_drawdown: float = 0.0
    max_observed_drawdown: float = 0.0
    risk_level: str = "low"  # "low", "medium", "high", "critical"
    action_required: str = ""  # "none", "reduce_position", "stop_trading"


class CVaRDrawdownController:
    """
    CVaR + Max Drawdown Risk Controller
    
    专注于实际风控效果，实时监控和响应风险。
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        risk_limits: Optional[RiskLimits] = None,
        lookback_window: int = 100,
        cvar_alpha: float = 0.05
    ):
        """
        初始化风控器
        
        Args:
            initial_capital: 初始资本
            risk_limits: 风控限制
            lookback_window: 回看窗口
            cvar_alpha: CVaR置信度
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        self.risk_limits = risk_limits or RiskLimits()
        self.lookback_window = lookback_window
        self.cvar_alpha = cvar_alpha
        
        # 收益历史
        self.returns_history = deque(maxlen=lookback_window)
        self.capital_history = deque(maxlen=lookback_window)
        self.capital_history.append(initial_capital)
        
        # 风险状态
        self.risk_status = RiskStatus()
        
        # 统计信息
        self.total_trades = 0
        self.risk_violations = 0
        self.warnings_issued = 0
    
    def update(
        self,
        portfolio_return: float,
        current_capital: Optional[float] = None
    ) -> RiskStatus:
        """
        更新风控状态
        
        Args:
            portfolio_return: 组合收益率
            current_capital: 当前资本（可选，如果不提供则从return计算）
            
        Returns:
            更新后的风险状态
        """
        # 更新资本
        if current_capital is not None:
            self.current_capital = current_capital
        else:
            self.current_capital *= (1 + portfolio_return)
        
        # 更新历史
        self.returns_history.append(portfolio_return)
        self.capital_history.append(self.current_capital)
        
        # 更新峰值
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # 计算CVaR
        cvar_95 = self._compute_cvar()
        
        # 计算当前回撤
        current_drawdown = (self.current_capital - self.peak_capital) / self.peak_capital
        
        # 更新最大回撤
        if current_drawdown < self.risk_status.max_observed_drawdown:
            self.risk_status.max_observed_drawdown = current_drawdown
        
        # 更新风险状态
        self.risk_status.cvar_95 = cvar_95
        self.risk_status.current_drawdown = current_drawdown
        
        # 评估风险等级
        self._evaluate_risk()
        
        self.total_trades += 1
        
        return self.risk_status
    
    def _compute_cvar(self) -> float:
        """
        计算CVaR (Conditional Value at Risk)
        
        CVaR = E[R | R <= VaR]
        """
        if len(self.returns_history) < 20:
            return 0.0
        
        returns_array = np.array(self.returns_history)
        
        # 计算VaR阈值
        var_threshold = np.quantile(returns_array, self.cvar_alpha)
        
        # CVaR是低于VaR的收益的期望值
        tail_returns = returns_array[returns_array <= var_threshold]
        
        if len(tail_returns) > 0:
            cvar = np.mean(tail_returns)
        else:
            cvar = var_threshold
        
        # 确保CVaR为负值（表示损失）
        return float(np.clip(cvar, -1.0, 0.0))
    
    def _evaluate_risk(self):
        """
        评估风险等级并决定需要的行动
        """
        cvar = self.risk_status.cvar_95
        drawdown = self.risk_status.current_drawdown
        
        # 检查是否超过硬限制
        cvar_violation = cvar < self.risk_limits.max_cvar_95
        drawdown_violation = drawdown < self.risk_limits.max_drawdown
        
        # 检查警告阈值
        cvar_warning = cvar < self.risk_limits.warning_cvar
        drawdown_warning = drawdown < self.risk_limits.warning_drawdown
        
        # 确定风险等级
        if cvar_violation or drawdown_violation:
            self.risk_status.risk_level = "critical"
            self.risk_status.is_safe = False
            self.risk_status.action_required = "stop_trading"
            self.risk_violations += 1
        elif cvar_warning or drawdown_warning:
            self.risk_status.risk_level = "high"
            self.risk_status.is_safe = False
            self.risk_status.action_required = "reduce_position"
            self.warnings_issued += 1
        elif cvar < self.risk_limits.warning_cvar * 0.8 or drawdown < self.risk_limits.warning_drawdown * 0.8:
            self.risk_status.risk_level = "medium"
            self.risk_status.is_safe = True
            self.risk_status.action_required = "monitor"
        else:
            self.risk_status.risk_level = "low"
            self.risk_status.is_safe = True
            self.risk_status.action_required = "none"
    
    def get_position_limit(self) -> float:
        """
        根据当前风险状态返回仓位限制
        
        Returns:
            仓位限制 (0.0-1.0)
        """
        if self.risk_status.risk_level == "critical":
            return 0.0  # 停止交易
        elif self.risk_status.risk_level == "high":
            return 0.3  # 大幅减仓
        elif self.risk_status.risk_level == "medium":
            return 0.6  # 适度减仓
        else:
            return 1.0  # 正常仓位
    
    def should_stop_trading(self) -> bool:
        """判断是否应该停止交易"""
        return self.risk_status.action_required == "stop_trading"
    
    def should_reduce_position(self) -> bool:
        """判断是否应该减仓"""
        return self.risk_status.action_required == "reduce_position"
    
    def get_risk_report(self) -> Dict:
        """
        获取风控报告
        
        Returns:
            风控报告字典
        """
        return {
            'risk_status': {
                'is_safe': self.risk_status.is_safe,
                'risk_level': self.risk_status.risk_level,
                'action_required': self.risk_status.action_required
            },
            'metrics': {
                'cvar_95': self.risk_status.cvar_95,
                'current_drawdown': self.risk_status.current_drawdown,
                'max_observed_drawdown': self.risk_status.max_observed_drawdown,
                'current_capital': self.current_capital,
                'peak_capital': self.peak_capital
            },
            'limits': {
                'max_cvar_95': self.risk_limits.max_cvar_95,
                'max_drawdown': self.risk_limits.max_drawdown,
                'warning_cvar': self.risk_limits.warning_cvar,
                'warning_drawdown': self.risk_limits.warning_drawdown
            },
            'statistics': {
                'total_trades': self.total_trades,
                'risk_violations': self.risk_violations,
                'warnings_issued': self.warnings_issued,
                'position_limit': self.get_position_limit()
            }
        }
    
    def reset(self):
        """重置风控器"""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.returns_history.clear()
        self.capital_history.clear()
        self.capital_history.append(self.initial_capital)
        self.risk_status = RiskStatus()
        self.total_trades = 0
        self.risk_violations = 0
        self.warnings_issued = 0
    
    def set_risk_limits(self, limits: RiskLimits):
        """设置风控限制"""
        self.risk_limits = limits


def create_risk_controller(
    initial_capital: float = 1_000_000.0,
    max_cvar_95: float = -0.05,
    max_drawdown: float = -0.15,
    **kwargs
) -> CVaRDrawdownController:
    """
    创建风控器的便捷函数
    
    Args:
        initial_capital: 初始资本
        max_cvar_95: 最大CVaR限制
        max_drawdown: 最大回撤限制
        **kwargs: 其他参数
        
    Returns:
        CVaRDrawdownController实例
    """
    risk_limits = RiskLimits(
        max_cvar_95=max_cvar_95,
        max_drawdown=max_drawdown
    )
    
    return CVaRDrawdownController(
        initial_capital=initial_capital,
        risk_limits=risk_limits,
        **kwargs
    )










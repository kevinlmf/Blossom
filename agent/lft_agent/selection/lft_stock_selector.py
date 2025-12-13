"""
LFT Stock Selector

专门为LFT Agent设计的选股模块，集成到组合构建流程中。

工作流程：
1. 从候选股票池中筛选股票（技术面 + 基本面）
2. 返回筛选后的股票列表和评分
3. LFT Agent基于筛选后的股票池构建组合
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import jax.numpy as jnp

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 可选导入stock_selection模块
try:
    from stock_selection import StockSelector, TechnicalSelector, FundamentalSelector
    STOCK_SELECTION_AVAILABLE = True
except ImportError:
    STOCK_SELECTION_AVAILABLE = False
    StockSelector = None
    TechnicalSelector = None
    FundamentalSelector = None


class LFTStockSelector:
    """
    LFT专用的选股器
    
    功能：
    1. 在LFT Agent构建组合前筛选股票池
    2. 集成技术面和基本面分析
    3. 返回筛选后的股票列表，供LFT Agent使用
    """

    def __init__(
        self,
        technical_weight: float = 0.5,
        fundamental_weight: float = 0.5,
        min_combined_score: float = 0.5,
        top_k: Optional[int] = None,
        enable_technical: bool = True,
        enable_fundamental: bool = True,
        rebalance_frequency: int = 20  # 每20个交易日重新选股
    ):
        """
        Args:
            technical_weight: 技术面权重
            fundamental_weight: 基本面权重
            min_combined_score: 最低综合评分阈值
            top_k: 返回前K只股票（如果None，返回所有符合条件的）
            enable_technical: 是否启用技术面选股
            enable_fundamental: 是否启用基本面选股
            rebalance_frequency: 选股重新平衡频率（交易日）
        """
        self.technical_weight = technical_weight
        self.fundamental_weight = fundamental_weight
        self.min_combined_score = min_combined_score
        self.top_k = top_k
        self.enable_technical = enable_technical
        self.enable_fundamental = enable_fundamental
        self.rebalance_frequency = rebalance_frequency
        
        # 创建选股器（如果stock_selection模块可用）
        if STOCK_SELECTION_AVAILABLE:
            technical_selector = TechnicalSelector() if enable_technical else None
            fundamental_selector = FundamentalSelector() if enable_fundamental else None
            
            self.selector = StockSelector(
                technical_weight=technical_weight,
                fundamental_weight=fundamental_weight,
                min_combined_score=min_combined_score,
                top_k=top_k,
                technical_selector=technical_selector,
                fundamental_selector=fundamental_selector
            )
        else:
            self.selector = None
            if enable_technical or enable_fundamental:
                print("Warning: stock_selection module not available. Stock selection disabled.")
        
        # 缓存选股结果
        self.last_selection_step = -1
        self.selected_symbols = []
        self.selection_scores = {}

    def select_stocks_for_lft(
        self,
        stock_data: Dict[str, Dict[str, np.ndarray]],
        symbols: List[str],
        current_step: int,
        force_rebalance: bool = False
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        为LFT Agent筛选股票
        
        Args:
            stock_data: 股票数据字典 {symbol: {'prices': array, 'volumes': array, ...}}
            symbols: 候选股票代码列表
            current_step: 当前时间步
            force_rebalance: 是否强制重新选股
        
        Returns:
            (筛选后的股票列表, 股票评分字典)
        """
        # 检查是否需要重新选股
        if (not force_rebalance and 
            self.last_selection_step >= 0 and
            current_step - self.last_selection_step < self.rebalance_frequency):
            # 使用缓存的选股结果
            return self.selected_symbols.copy(), self.selection_scores.copy()
        
        # 执行选股
        if self.selector is None:
            # 如果选股器不可用，返回所有股票
            return symbols, {sym: 0.5 for sym in symbols}
        
        try:
            combined_scores = self.selector.select_stocks(
                stock_data=stock_data,
                symbols=symbols,
                min_score=self.min_combined_score
            )
            
            # 提取股票列表和评分
            selected_symbols = [score.symbol for score in combined_scores]
            selection_scores = {
                score.symbol: score.total_score 
                for score in combined_scores
            }
            
            # 更新缓存
            self.last_selection_step = current_step
            self.selected_symbols = selected_symbols
            self.selection_scores = selection_scores
            
            return selected_symbols, selection_scores
            
        except Exception as e:
            print(f"Warning: Stock selection failed: {e}")
            print(f"Falling back to all symbols")
            # 如果选股失败，返回所有股票
            return symbols, {sym: 0.5 for sym in symbols}

    def filter_stock_data(
        self,
        stock_data: Dict[str, Dict[str, np.ndarray]],
        selected_symbols: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        根据选股结果过滤股票数据
        
        Args:
            stock_data: 原始股票数据
            selected_symbols: 筛选后的股票列表
        
        Returns:
            过滤后的股票数据
        """
        return {
            symbol: data 
            for symbol, data in stock_data.items() 
            if symbol in selected_symbols
        }

    def get_selection_statistics(self) -> Dict[str, any]:
        """获取选股统计信息"""
        return {
            'last_selection_step': self.last_selection_step,
            'num_selected': len(self.selected_symbols),
            'selected_symbols': self.selected_symbols.copy(),
            'selection_scores': self.selection_scores.copy(),
            'rebalance_frequency': self.rebalance_frequency
        }

    def reset(self):
        """重置选股器状态"""
        self.last_selection_step = -1
        self.selected_symbols = []
        self.selection_scores = {}


def integrate_with_daily_data_loader(
    data_loader,
    stock_selector: LFTStockSelector,
    stock_data: Dict[str, Dict[str, np.ndarray]],
    symbols: List[str],
    current_step: int
) -> Tuple[Dict[str, jnp.ndarray], List[str]]:
    """
    将选股器集成到DailyDataLoader中
    
    这是一个辅助函数，用于在数据加载时进行选股
    
    Args:
        data_loader: DailyDataLoader实例
        stock_selector: LFTStockSelector实例
        stock_data: 股票数据字典
        symbols: 候选股票列表
        current_step: 当前时间步
    
    Returns:
        (过滤后的数据, 选中的股票列表)
    """
    # 执行选股
    selected_symbols, scores = stock_selector.select_stocks_for_lft(
        stock_data=stock_data,
        symbols=symbols,
        current_step=current_step
    )
    
    # 过滤数据
    filtered_data = stock_selector.filter_stock_data(stock_data, selected_symbols)
    
    # 更新data_loader的num_assets
    data_loader.num_assets = len(selected_symbols)
    
    return filtered_data, selected_symbols





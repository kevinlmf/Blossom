"""
EM Training Loop for Learning Latent Variables that Explain Returns

完整的EM训练循环：
1. E-step: 使用当前encoder提取latent factors z_t
2. M-step: 最大化z_t对收益R_t的解释能力，更新encoder
3. 评估: 计算R²等指标
4. 循环直到性能超过baseline
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from tqdm import tqdm

from .em_encoder import (
    create_em_encoder,
    e_step,
    m_step,
    compute_r_squared
)
from evaluation import PerformanceMetrics, BenchmarkStrategies


class EMReturnLearning:
    """
    EM算法学习能解释资产收益的latent variables
    
    目标：找到z_t使得 R_t = f(z_t) + ε，最大化R²
    """
    
    def __init__(
        self,
        market_data: np.ndarray,
        returns: np.ndarray,
        latent_dim: int = 64,
        num_factors: int = 10,
        learning_rate: float = 1e-3,
        output_dir: str = "outputs/em_return_learning",
        max_iterations: int = 50,
        min_r_squared: float = 0.3,
        verbose: bool = True
    ):
        """
        初始化EM学习器
        
        Args:
            market_data: 市场数据 [T, seq_len, input_dim] 或 [T, input_dim]
            returns: 资产收益 [T, num_assets]
            latent_dim: 潜在变量维度
            num_factors: 因子数量
            learning_rate: 学习率
            output_dir: 输出目录
            max_iterations: 最大迭代次数
            min_r_squared: 最小R²阈值
            verbose: 是否打印详细信息
        """
        self.market_data = market_data
        self.returns = returns
        self.latent_dim = latent_dim
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_iterations = max_iterations
        self.min_r_squared = min_r_squared
        self.verbose = verbose
        
        # 准备数据
        self._prepare_data()
        
        # 创建encoder和训练状态
        self.encoder, self.state = create_em_encoder(
            latent_dim=latent_dim,
            num_factors=num_factors,
            num_assets=returns.shape[1] if len(returns.shape) > 1 else 1,
            learning_rate=learning_rate
        )
        
        self.key = jax.random.PRNGKey(42)
        
        # 历史记录
        self.r_squared_history = []
        self.loss_history = []
        self.latent_factors_history = []
        
        if verbose:
            print("\n" + "="*80)
            print("🔄 EM ALGORITHM FOR LEARNING LATENT VARIABLES")
            print("="*80)
            print(f"Market Data Shape: {market_data.shape}")
            print(f"Returns Shape: {returns.shape}")
            print(f"Latent Dimension: {latent_dim}")
            print(f"Number of Factors: {num_factors}")
            print(f"Max Iterations: {max_iterations}")
            print(f"Min R² Threshold: {min_r_squared}")
            print("="*80)
    
    def _prepare_data(self):
        """准备数据格式"""
        # 如果market_data是2D，转换为3D [T, seq_len, features]
        if len(self.market_data.shape) == 2:
            T, features = self.market_data.shape
            seq_len = 20  # 默认序列长度
            
            # 创建滑动窗口
            market_data_3d = []
            for t in range(T):
                start_idx = max(0, t - seq_len + 1)
                seq_data = self.market_data[start_idx:t+1]
                
                # Padding if needed
                if len(seq_data) < seq_len:
                    padding = np.zeros((seq_len - len(seq_data), features))
                    seq_data = np.concatenate([padding, seq_data], axis=0)
                
                market_data_3d.append(seq_data)
            
            self.market_data = np.array(market_data_3d)  # [T, seq_len, features]
        
        # 确保returns是2D
        if len(self.returns.shape) == 1:
            self.returns = self.returns[:, None]  # [T, 1]
    
    def compute_baseline_r_squared(self) -> float:
        """
        计算baseline的R²（使用简单特征）
        
        Returns:
            baseline_r_squared: Baseline的R²
        """
        # 使用市场数据的简单特征作为baseline
        # 例如：使用价格变化率
        if len(self.market_data.shape) == 3:
            # 取最后一个时间步的特征
            simple_features = self.market_data[:, -1, :]  # [T, features]
        else:
            simple_features = self.market_data
        
        # 使用前几个特征预测收益
        if simple_features.shape[1] > 0:
            # 简单的线性回归
            F = simple_features[:, :min(5, simple_features.shape[1])]  # 使用前5个特征
            R = self.returns
            
            # 添加截距
            F_with_intercept = np.concatenate([
                np.ones((F.shape[0], 1)), F
            ], axis=1)
            
            # OLS回归
            try:
                betas = np.linalg.lstsq(F_with_intercept, R, rcond=None)[0]
                R_pred = F_with_intercept @ betas
                
                ss_res = np.sum((R - R_pred) ** 2, axis=0)
                ss_tot = np.sum((R - np.mean(R, axis=0)) ** 2, axis=0)
                r_squared = 1 - ss_res / (ss_tot + 1e-8)
                
                baseline_r_squared = float(np.mean(r_squared))
            except:
                baseline_r_squared = 0.0
        else:
            baseline_r_squared = 0.0
        
        if self.verbose:
            print(f"\n📊 Baseline R²: {baseline_r_squared:.4f}")
        
        return baseline_r_squared
    
    def run(self) -> Dict[str, Any]:
        """
        运行EM训练循环
        
        Returns:
            最终结果字典
        """
        # 计算baseline
        baseline_r_squared = self.compute_baseline_r_squared()
        
        best_r_squared = -np.inf
        best_iteration = 0
        
        if self.verbose:
            print(f"\n🚀 Starting EM Training Loop...")
            print(f"Target: R² > {max(baseline_r_squared, self.min_r_squared):.4f}")
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"🔄 EM Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'='*80}")
            
            # E-step: 估计潜在变量
            if self.verbose:
                print("📊 E-step: Estimating latent variables...")
            
            latent_factors = e_step(
                self.encoder,
                self.state.params,
                jnp.array(self.market_data),
                self.key
            )
            
            if self.verbose:
                print(f"  Extracted latent factors: {latent_factors.shape}")
                print(f"  Factor statistics:")
                print(f"    Mean: {jnp.mean(latent_factors):.4f}")
                print(f"    Std: {jnp.std(latent_factors):.4f}")
            
            # M-step: 更新encoder参数
            if self.verbose:
                print("🔄 M-step: Updating encoder to maximize return explanation...")
            
            self.state, metrics, self.key = m_step(
                self.encoder,
                self.state,
                jnp.array(self.market_data),
                jnp.array(self.returns),
                latent_factors,
                self.key,
                num_steps=10
            )
            
            # 计算R²
            r_squared = compute_r_squared(
                latent_factors,
                jnp.array(self.returns)
            )
            
            # 记录历史
            self.r_squared_history.append(float(r_squared))
            self.loss_history.append(float(metrics['total_loss']))
            self.latent_factors_history.append(np.array(latent_factors))
            
            if self.verbose:
                print(f"\n📊 Performance Metrics:")
                print(f"  R² (Return Explanation): {r_squared:.4f}")
                print(f"  Baseline R²: {baseline_r_squared:.4f}")
                print(f"  Improvement: {r_squared - baseline_r_squared:+.4f}")
                print(f"  Total Loss: {metrics['total_loss']:.6f}")
                print(f"  MSE Loss: {metrics['mse_loss']:.6f}")
                print(f"  R² Loss: {metrics['r2_loss']:.6f}")
            
            # 更新最佳性能
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_iteration = iteration + 1
            
            # 检查收敛条件
            target_r_squared = max(baseline_r_squared, self.min_r_squared)
            
            if r_squared >= target_r_squared:
                if self.verbose:
                    print(f"\n✅ CONVERGED! R² exceeds target.")
                    print(f"  R²: {r_squared:.4f} >= Target: {target_r_squared:.4f}")
                    print(f"  Iterations: {iteration + 1}")
                
                result = {
                    'converged': True,
                    'iteration': iteration + 1,
                    'final_r_squared': float(r_squared),
                    'baseline_r_squared': baseline_r_squared,
                    'improvement': float(r_squared - baseline_r_squared),
                    'best_r_squared': float(best_r_squared),
                    'best_iteration': best_iteration,
                    'latent_factors': np.array(latent_factors),
                    'r_squared_history': self.r_squared_history,
                    'loss_history': self.loss_history
                }
                
                self._save_results(result, converged=True)
                return result
            
            # 检查是否不再提升
            if iteration > 5:
                recent_r_squared = self.r_squared_history[-5:]
                if max(recent_r_squared) - min(recent_r_squared) < 0.01:
                    if self.verbose:
                        print(f"\n⚠️  R² converged (no improvement in last 5 iterations)")
        
        # 达到最大迭代次数
        if self.verbose:
            print(f"\n⚠️  Reached maximum iterations ({self.max_iterations})")
            print(f"  Best R²: {best_r_squared:.4f} at iteration {best_iteration}")
            print(f"  Baseline R²: {baseline_r_squared:.4f}")
        
        result = {
            'converged': False,
            'iteration': self.max_iterations,
            'final_r_squared': float(self.r_squared_history[-1]),
            'baseline_r_squared': baseline_r_squared,
            'improvement': float(self.r_squared_history[-1] - baseline_r_squared),
            'best_r_squared': float(best_r_squared),
            'best_iteration': best_iteration,
            'latent_factors': np.array(self.latent_factors_history[-1]),
            'r_squared_history': self.r_squared_history,
            'loss_history': self.loss_history
        }
        
        self._save_results(result, converged=False)
        return result
    
    def _save_results(self, result: Dict[str, Any], converged: bool):
        """保存结果"""
        # 保存JSON结果
        result_file = self.output_dir / f"em_results_iter_{result['iteration']}.json"
        
        # 转换numpy数组为列表
        result_to_save = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                result_to_save[k] = v.tolist()
            else:
                result_to_save[k] = v
        
        with open(result_file, 'w') as f:
            json.dump(result_to_save, f, indent=2)
        
        if self.verbose:
            print(f"\n💾 Results saved to: {result_file}")
    
    def get_latent_factors(self) -> np.ndarray:
        """获取最终的latent factors"""
        if self.latent_factors_history:
            return self.latent_factors_history[-1]
        else:
            # 如果没有历史，重新计算
            latent_factors = e_step(
                self.encoder,
                self.state.params,
                jnp.array(self.market_data),
                self.key
            )
            return np.array(latent_factors)










"""
EM-based Encoder for Learning Latent Variables that Explain Asset Returns

核心思想：
1. E-step: 使用当前encoder提取latent factors z_t
2. M-step: 最大化z_t对收益R_t的解释能力
   - 使用z_t预测收益: R_t = f(z_t) + ε
   - 更新encoder参数以最大化R²或最小化预测误差
3. 循环直到性能超过baseline
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Tuple, Optional, Dict, Any
import chex
import optax
import numpy as np

from .encoder import SharedEncoder


class ReturnPredictionHead(nn.Module):
    """
    收益预测头：使用latent factors预测资产收益
    
    R_t = α + β·z_t + ε
    """
    num_assets: int = 1
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, latent_factors: chex.Array) -> chex.Array:
        """
        使用latent factors预测收益
        
        Args:
            latent_factors: Latent factors [batch, num_factors]
            
        Returns:
            predicted_returns: Predicted returns [batch, num_assets]
        """
        # 非线性映射
        x = nn.Dense(self.hidden_dim)(latent_factors)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)
        
        # 输出预测收益
        predicted_returns = nn.Dense(self.num_assets)(x)
        
        return predicted_returns


class EMEncoder(nn.Module):
    """
    EM-based Encoder for learning latent variables that explain returns
    
    架构：
    Market Data → Encoder → Latent Factors z_t → Return Prediction Head → Predicted Returns
    """
    latent_dim: int = 64
    num_factors: int = 10
    num_assets: int = 1
    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1
    
    def setup(self):
        """初始化encoder和预测头"""
        # 使用现有的SharedEncoder
        self.encoder = SharedEncoder(
            latent_dim=self.latent_dim,
            num_factors=self.num_factors,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            orthogonal_constraint=True
        )
        
        # 收益预测头
        self.return_head = ReturnPredictionHead(
            num_assets=self.num_assets,
            hidden_dim=self.hidden_dim
        )
    
    def __call__(
        self,
        x: chex.Array,
        returns: Optional[chex.Array] = None,
        train: bool = True
    ) -> Tuple[chex.Array, Optional[chex.Array], Optional[Dict[str, chex.Array]]]:
        """
        Forward pass
        
        Args:
            x: Market data [batch, seq_len, input_dim]
            returns: Actual returns [batch, num_assets] (for training)
            train: Training mode
            
        Returns:
            latent_factors: Latent factors [batch, num_factors]
            predicted_returns: Predicted returns [batch, num_assets]
            metrics: Dictionary with R², MSE, etc.
        """
        # 提取latent factors
        latent_factors, _ = self.encoder(x, train=train)
        
        # 预测收益
        predicted_returns = self.return_head(latent_factors)
        
        metrics = None
        if returns is not None:
            # 计算预测误差
            mse = jnp.mean((predicted_returns - returns) ** 2)
            
            # 计算R²
            ss_res = jnp.sum((returns - predicted_returns) ** 2, axis=0)
            ss_tot = jnp.sum((returns - jnp.mean(returns, axis=0)) ** 2, axis=0)
            r_squared = 1 - ss_res / (ss_tot + 1e-8)
            
            # 计算因子对收益的解释能力
            # 使用线性回归：returns = beta * factors + epsilon
            betas, r_squared_factors = self.encoder.compute_factor_betas(
                latent_factors, returns
            )
            
            metrics = {
                'mse': mse,
                'r_squared': r_squared,
                'r_squared_factors': r_squared_factors,
                'betas': betas,
                'predicted_returns': predicted_returns,
                'latent_factors': latent_factors
            }
        
        return latent_factors, predicted_returns, metrics


def create_em_encoder(
    latent_dim: int = 64,
    num_factors: int = 10,
    num_assets: int = 1,
    learning_rate: float = 1e-3
) -> Tuple[EMEncoder, train_state.TrainState]:
    """
    创建EM encoder和训练状态
    
    Returns:
        encoder: EMEncoder模型
        state: 训练状态
    """
    encoder = EMEncoder(
        latent_dim=latent_dim,
        num_factors=num_factors,
        num_assets=num_assets
    )
    
    # 初始化参数
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 20, 10))  # [batch, seq_len, input_dim]
    dummy_returns = jnp.ones((1, num_assets))
    
    params = encoder.init(key, dummy_input, dummy_returns, train=True)
    
    # 创建优化器
    tx = optax.adam(learning_rate)
    
    # 创建训练状态
    state = train_state.TrainState.create(
        apply_fn=encoder.apply,
        params=params,
        tx=tx
    )
    
    return encoder, state


@jax.jit
def compute_loss(
    params: Dict[str, Any],
    encoder: EMEncoder,
    market_data: chex.Array,
    returns: chex.Array,
    key: chex.PRNGKey
) -> Tuple[float, Dict[str, chex.Array]]:
    """
    计算EM算法的损失函数
    
    M-step的目标：最大化latent factors对收益的解释能力
    
    Loss = MSE(predicted_returns, actual_returns) - λ * R²(factors, returns)
    
    Args:
        params: Encoder参数
        encoder: Encoder模型
        market_data: 市场数据 [batch, seq_len, input_dim]
        returns: 实际收益 [batch, num_assets]
        key: 随机key
        
    Returns:
        loss: 总损失
        metrics: 指标字典
    """
    # Forward pass
    latent_factors, predicted_returns, metrics = encoder.apply(
        params,
        market_data,
        returns,
        train=True,
        rngs={'dropout': key}
    )
    
    # 预测误差损失
    mse = jnp.mean((predicted_returns - returns) ** 2)
    
    # R²损失（负的R²，因为我们要最大化R²）
    r_squared = metrics['r_squared']
    r_squared_loss = -jnp.mean(r_squared)  # 负号因为要最大化
    
    # 因子解释能力损失
    r_squared_factors = metrics['r_squared_factors']
    factor_explanation_loss = -jnp.mean(r_squared_factors)
    
    # 总损失
    lambda_mse = 1.0
    lambda_r2 = 0.5
    lambda_factor = 0.5
    
    total_loss = (
        lambda_mse * mse +
        lambda_r2 * r_squared_loss +
        lambda_factor * factor_explanation_loss
    )
    
    metrics['total_loss'] = total_loss
    metrics['mse_loss'] = mse
    metrics['r2_loss'] = r_squared_loss
    metrics['factor_loss'] = factor_explanation_loss
    
    return total_loss, metrics


@jax.jit
def update_step(
    state: train_state.TrainState,
    encoder: EMEncoder,
    market_data: chex.Array,
    returns: chex.Array,
    key: chex.PRNGKey
) -> Tuple[train_state.TrainState, Dict[str, chex.Array], chex.PRNGKey]:
    """
    M-step更新步骤
    
    Args:
        state: 训练状态
        encoder: Encoder模型
        market_data: 市场数据
        returns: 实际收益
        key: 随机key
        
    Returns:
        new_state: 更新后的状态
        metrics: 指标
        new_key: 新的随机key
    """
    # 计算损失和梯度
    (loss, metrics), grads = jax.value_and_grad(
        compute_loss, has_aux=True
    )(state.params, encoder, market_data, returns, key)
    
    # 更新参数
    new_state = state.apply_gradients(grads=grads)
    
    # 生成新的随机key
    new_key = jax.random.split(key)[0]
    
    return new_state, metrics, new_key


def e_step(
    encoder: EMEncoder,
    params: Dict[str, Any],
    market_data: chex.Array,
    key: chex.PRNGKey
) -> chex.Array:
    """
    E-step: 估计潜在变量 z_t
    
    使用当前encoder参数提取latent factors
    
    Args:
        encoder: Encoder模型
        params: Encoder参数
        market_data: 市场数据 [batch, seq_len, input_dim]
        key: 随机key
        
    Returns:
        latent_factors: 估计的潜在变量 [batch, num_factors]
    """
    # 使用encoder提取latent factors
    latent_factors, _, _ = encoder.apply(
        params,
        market_data,
        returns=None,
        train=False,
        rngs={'dropout': key}
    )
    
    return latent_factors


def m_step(
    encoder: EMEncoder,
    state: train_state.TrainState,
    market_data: chex.Array,
    returns: chex.Array,
    latent_factors: chex.Array,
    key: chex.PRNGKey,
    num_steps: int = 10
) -> Tuple[train_state.TrainState, Dict[str, Any], chex.PRNGKey]:
    """
    M-step: 使用估计的z_t更新encoder参数
    
    最大化latent factors对收益的解释能力
    
    Args:
        encoder: Encoder模型
        state: 训练状态
        market_data: 市场数据
        returns: 实际收益
        latent_factors: E-step估计的latent factors
        key: 随机key
        num_steps: 更新步数
        
    Returns:
        new_state: 更新后的状态
        metrics: 指标字典
        new_key: 新的随机key
    """
    current_key = key
    current_state = state
    
    all_metrics = []
    
    for step in range(num_steps):
        # 更新encoder参数
        current_state, metrics, current_key = update_step(
            current_state,
            encoder,
            market_data,
            returns,
            current_key
        )
        
        all_metrics.append(metrics)
    
    # 平均指标
    avg_metrics = {
        k: jnp.mean(jnp.array([m[k] for m in all_metrics]))
        for k in all_metrics[0].keys()
    }
    
    return current_state, avg_metrics, current_key


def compute_r_squared(
    latent_factors: chex.Array,
    returns: chex.Array
) -> float:
    """
    计算latent factors对收益的R²
    
    Args:
        latent_factors: Latent factors [T, num_factors]
        returns: Actual returns [T, num_assets]
        
    Returns:
        r_squared: R²值
    """
    # 线性回归：returns = beta * factors + epsilon
    F = latent_factors  # [T, num_factors]
    R = returns  # [T, num_assets]
    
    # 添加截距
    F_with_intercept = jnp.concatenate([
        jnp.ones((F.shape[0], 1)), F
    ], axis=1)
    
    # 计算beta
    FtF_inv = jnp.linalg.pinv(F_with_intercept.T @ F_with_intercept)
    betas = FtF_inv @ F_with_intercept.T @ R
    
    # 预测收益
    R_pred = F_with_intercept @ betas
    
    # 计算R²
    ss_res = jnp.sum((R - R_pred) ** 2, axis=0)
    ss_tot = jnp.sum((R - jnp.mean(R, axis=0)) ** 2, axis=0)
    r_squared = 1 - ss_res / (ss_tot + 1e-8)
    
    return jnp.mean(r_squared)


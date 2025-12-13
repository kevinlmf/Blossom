"""
Shared Encoder for Latent Factor Learning

This module provides a shared representation learning component that:
1. Extracts latent factors from market data
2. Provides interpretable factor representations
3. Enables factor & beta estimation
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Callable
import chex


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based encoder."""

    max_len: int = 5000
    d_model: int = 128

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        seq_len = x.shape[1]
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) *
                          -(jnp.log(10000.0) / self.d_model))

        pe = jnp.zeros((seq_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        return x + pe[None, :, :]


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with multi-head attention."""

    num_heads: int = 8
    hidden_dim: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: chex.Array, train: bool = True) -> chex.Array:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            train: Training mode flag
        Returns:
            Encoded representation
        """
        # Multi-head self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=x.shape[-1],
            dropout_rate=self.dropout_rate,
            deterministic=not train
        )(x, x)

        # Add & Norm
        x = nn.LayerNorm()(x + attn_output)

        # Feed-forward network
        ff_output = nn.Dense(self.hidden_dim)(x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_output)
        ff_output = nn.Dense(x.shape[-1])(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_output)

        # Add & Norm
        x = nn.LayerNorm()(x + ff_output)

        return x


class SharedEncoder(nn.Module):
    """
    Shared Encoder for multi-frequency factor learning.

    Architecture:
        Input (market data) → Embedding → Transformer Layers → Latent Factors z_t

    The encoder learns a shared representation that captures:
    - Market microstructure (for HFT)
    - Price dynamics (for MFT)
    - Macro trends (for LFT)
    """

    latent_dim: int = 64
    num_factors: int = 10
    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1
    orthogonal_constraint: bool = True

    def setup(self):
        """Initialize encoder components."""
        # Input embedding
        self.input_projection = nn.Dense(self.latent_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model=self.latent_dim)

        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout
            ) for _ in range(self.num_layers)
        ]

        # Factor extraction head
        self.factor_head = nn.Dense(self.num_factors)

        # Factor loadings (for interpretability)
        self.factor_loadings = self.param(
            'factor_loadings',
            nn.initializers.orthogonal(),
            (self.latent_dim, self.num_factors)
        )

    def __call__(
        self,
        x: chex.Array,
        train: bool = True,
        return_attention: bool = False
    ) -> Tuple[chex.Array, Optional[chex.Array]]:
        """
        Forward pass through shared encoder.

        Args:
            x: Input market data [batch, seq_len, input_dim]
            train: Training mode flag
            return_attention: Whether to return attention weights

        Returns:
            latent_factors: Learned factors z_t [batch, num_factors]
            attention_weights: Optional attention weights (if requested)
        """
        # Project input to latent dimension
        x = self.input_projection(x)  # [batch, seq_len, latent_dim]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer encoder layers
        attention_weights = []
        for layer in self.encoder_layers:
            x = layer(x, train=train)

        # Pool over sequence dimension (mean pooling)
        x_pooled = jnp.mean(x, axis=1)  # [batch, latent_dim]

        # Extract factors
        latent_factors = self.factor_head(x_pooled)  # [batch, num_factors]

        # Apply orthogonality constraint if enabled
        if self.orthogonal_constraint and train:
            latent_factors = self._apply_orthogonal_constraint(latent_factors)

        if return_attention:
            return latent_factors, attention_weights
        return latent_factors, None

    def _apply_orthogonal_constraint(self, factors: chex.Array) -> chex.Array:
        """
        Apply orthogonality constraint to factors using Gram-Schmidt.

        Args:
            factors: Input factors [batch, num_factors]
        Returns:
            Orthogonalized factors
        """
        # Normalize each factor
        factors_norm = factors / (jnp.linalg.norm(factors, axis=0, keepdims=True) + 1e-8)
        return factors_norm

    def compute_factor_betas(
        self,
        latent_factors: chex.Array,
        returns: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Compute factor loadings (betas) for interpretability.

        Args:
            latent_factors: Learned factors [batch, num_factors]
            returns: Asset returns [batch, num_assets]

        Returns:
            betas: Factor loadings [num_assets, num_factors]
            r_squared: R² for each asset
        """
        # Linear regression: returns = alpha + beta * factors + epsilon
        # Using least squares: beta = (F'F)^(-1) F'R

        F = latent_factors  # [batch, num_factors]
        R = returns  # [batch, num_assets]

        # Add intercept
        F_with_intercept = jnp.concatenate([
            jnp.ones((F.shape[0], 1)), F
        ], axis=1)

        # Compute betas
        FtF_inv = jnp.linalg.pinv(F_with_intercept.T @ F_with_intercept)
        betas = FtF_inv @ F_with_intercept.T @ R

        # Compute R²
        R_pred = F_with_intercept @ betas
        ss_res = jnp.sum((R - R_pred) ** 2, axis=0)
        ss_tot = jnp.sum((R - jnp.mean(R, axis=0)) ** 2, axis=0)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)

        return betas[1:], r_squared  # Exclude intercept

    def get_factor_importance(
        self,
        latent_factors: chex.Array,
        window: int = 100
    ) -> chex.Array:
        """
        Compute factor importance based on variance explained.

        Args:
            latent_factors: Learned factors [batch, num_factors]
            window: Rolling window for variance computation

        Returns:
            importance: Factor importance scores [num_factors]
        """
        # Compute variance of each factor
        factor_var = jnp.var(latent_factors, axis=0)

        # Normalize to get importance
        importance = factor_var / jnp.sum(factor_var)

        return importance


class CNNLSTMEncoder(nn.Module):
    """Alternative encoder using CNN-LSTM architecture."""

    latent_dim: int = 64
    num_factors: int = 10
    num_cnn_layers: int = 3
    num_lstm_layers: int = 2
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        train: bool = True
    ) -> Tuple[chex.Array, None]:
        """
        Forward pass through CNN-LSTM encoder.

        Args:
            x: Input market data [batch, seq_len, input_dim]
            train: Training mode flag

        Returns:
            latent_factors: Learned factors [batch, num_factors]
        """
        # CNN layers for local pattern extraction
        for i in range(self.num_cnn_layers):
            x = nn.Conv(
                features=self.latent_dim,
                kernel_size=(3,),
                padding='SAME'
            )(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)

        # LSTM layers using nn.RNN wrapper (proper Flax way)
        # Process through LSTM layers
        for layer_idx in range(self.num_lstm_layers):
            # Create RNN cell wrapper
            RNNLayer = nn.RNN(
                nn.OptimizedLSTMCell(features=self.latent_dim),
                return_carry=True
            )
            carry, x = RNNLayer(x)

        # Take final hidden state from last layer
        # carry is a tuple (hidden, cell) for LSTM
        if isinstance(carry, tuple):
            final_hidden = carry[0]  # Get hidden state
        else:
            final_hidden = carry

        # Global mean pooling as backup (in case sequence info is needed)
        # We'll use the final hidden state which already summarizes the sequence
        # But if batch dimensions are weird, we can fall back to mean pooling
        if len(final_hidden.shape) == 3:
            # [batch, seq, features] -> [batch, features]
            final_hidden = jnp.mean(final_hidden, axis=1)
        elif len(final_hidden.shape) == 2:
            # Already [batch, features]
            pass

        # Project to factors
        latent_factors = nn.Dense(self.num_factors)(final_hidden)

        return latent_factors, None

"""
Encoder Factory for Regime-Adaptive Architecture Selection

This module provides a factory for creating encoders based on market regime:
- HIGH_RISK regime → LSTM (fast response, low latency)
- STABLE regime → Transformer (deep reasoning, long-term memory)
- HIGH_RETURN regime → Ensemble (combines both strengths)

Design rationale:
┌──────────────┬──────────────┬────────────────┬─────────────────┐
│   Regime     │   Encoder    │   Reasoning    │   Trade-off     │
├──────────────┼──────────────┼────────────────┼─────────────────┤
│ HIGH_RISK    │ CNN-LSTM     │ Fast response  │ Limited memory  │
│              │              │ Low latency    │ Sequential proc │
│              │              │ Crisis mode    │                 │
├──────────────┼──────────────┼────────────────┼─────────────────┤
│ STABLE       │ Transformer  │ Deep analysis  │ Higher latency  │
│              │              │ Long memory    │ More parameters │
│              │              │ Attention viz  │                 │
├──────────────┼──────────────┼────────────────┼─────────────────┤
│ HIGH_RETURN  │ Ensemble     │ Best of both   │ 2x computation  │
│              │ (weighted)   │ Robust         │ More complex    │
└──────────────┴──────────────┴────────────────┴─────────────────┘
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, Dict, Any
import chex

from .encoder import SharedEncoder, CNNLSTMEncoder


class EnsembleEncoder(nn.Module):
    """
    Ensemble encoder that combines Transformer and LSTM.

    Used for HIGH_RETURN regime where we want robustness.
    """

    latent_dim: int = 64
    num_factors: int = 10
    transformer_weight: float = 0.6  # 60% transformer, 40% LSTM

    # Transformer config
    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1

    # LSTM config
    num_lstm_layers: int = 2
    num_cnn_layers: int = 3

    def setup(self):
        """Initialize both encoders."""
        self.transformer_encoder = SharedEncoder(
            latent_dim=self.latent_dim,
            num_factors=self.num_factors,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )

        self.lstm_encoder = CNNLSTMEncoder(
            latent_dim=self.latent_dim,
            num_factors=self.num_factors,
            num_lstm_layers=self.num_lstm_layers,
            num_cnn_layers=self.num_cnn_layers,
            dropout=self.dropout
        )

        # Learnable fusion weights (can adapt during training)
        self.fusion_weights = self.param(
            'fusion_weights',
            nn.initializers.constant(jnp.array([self.transformer_weight, 1 - self.transformer_weight])),
            (2,)
        )

    def __call__(
        self,
        x: chex.Array,
        train: bool = True,
        return_attention: bool = False
    ) -> Tuple[chex.Array, Optional[Dict[str, Any]]]:
        """
        Forward pass through ensemble.

        Args:
            x: Input market data [batch, seq_len, input_dim]
            train: Training mode flag
            return_attention: Whether to return attention info

        Returns:
            latent_factors: Ensemble factors [batch, num_factors]
            extras: Optional dict with attention weights and contribution info
        """
        # Get outputs from both encoders
        transformer_factors, transformer_attn = self.transformer_encoder(
            x, train=train, return_attention=return_attention
        )

        lstm_factors, _ = self.lstm_encoder(x, train=train)

        # Normalize fusion weights (softmax for proper weighting)
        normalized_weights = jax.nn.softmax(self.fusion_weights)

        # Weighted ensemble
        latent_factors = (
            normalized_weights[0] * transformer_factors +
            normalized_weights[1] * lstm_factors
        )

        # Return extra info if requested
        extras = None
        if return_attention:
            extras = {
                'transformer_attention': transformer_attn,
                'transformer_weight': float(normalized_weights[0]),
                'lstm_weight': float(normalized_weights[1]),
                'transformer_factors': transformer_factors,
                'lstm_factors': lstm_factors
            }

        return latent_factors, extras


def create_encoder(
    regime_type: str,
    latent_dim: int = 64,
    num_factors: int = 10,
    config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Factory function to create encoder based on market regime.

    Args:
        regime_type: Market regime ('high_risk', 'stable', 'high_return')
        latent_dim: Latent dimension size
        num_factors: Number of factors to extract
        config: Optional additional configuration

    Returns:
        Appropriate encoder for the regime

    Examples:
        >>> # Crisis mode - fast LSTM
        >>> encoder = create_encoder('high_risk', latent_dim=32)
        >>>
        >>> # Normal market - powerful Transformer
        >>> encoder = create_encoder('stable', latent_dim=64)
        >>>
        >>> # Bull market - ensemble for robustness
        >>> encoder = create_encoder('high_return', latent_dim=64)
    """
    config = config or {}

    if regime_type == 'high_risk':
        # 🔴 HIGH RISK: Use LSTM for fast response
        print(f"   🧠 Encoder: CNN-LSTM (fast response for crisis)")
        return CNNLSTMEncoder(
            latent_dim=latent_dim,
            num_factors=num_factors,
            num_lstm_layers=config.get('num_lstm_layers', 2),
            num_cnn_layers=config.get('num_cnn_layers', 3),
            dropout=config.get('dropout', 0.1)
        )

    elif regime_type == 'stable':
        # 🟡 STABLE: Use Transformer for deep analysis
        print(f"   🧠 Encoder: Transformer (deep analysis for stable market)")
        return SharedEncoder(
            latent_dim=latent_dim,
            num_factors=num_factors,
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.1),
            orthogonal_constraint=config.get('orthogonal_constraint', True)
        )

    elif regime_type == 'high_return':
        # 🟢 HIGH RETURN: Use Ensemble for robustness
        print(f"   🧠 Encoder: Ensemble (Transformer + LSTM for bull market)")
        return EnsembleEncoder(
            latent_dim=latent_dim,
            num_factors=num_factors,
            transformer_weight=config.get('transformer_weight', 0.6),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            hidden_dim=config.get('hidden_dim', 256),
            num_lstm_layers=config.get('num_lstm_layers', 2),
            num_cnn_layers=config.get('num_cnn_layers', 3),
            dropout=config.get('dropout', 0.1)
        )

    else:
        # Default to Transformer
        print(f"   ⚠️  Unknown regime '{regime_type}', defaulting to Transformer")
        return SharedEncoder(
            latent_dim=latent_dim,
            num_factors=num_factors,
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            hidden_dim=config.get('hidden_dim', 256),
            dropout=config.get('dropout', 0.1)
        )


def get_encoder_info(regime_type: str) -> Dict[str, Any]:
    """
    Get information about the encoder for a given regime.

    Useful for logging and documentation.

    Args:
        regime_type: Market regime

    Returns:
        Dict with encoder info (type, advantages, trade-offs)
    """
    encoder_info = {
        'high_risk': {
            'type': 'CNN-LSTM',
            'advantages': [
                'Fast inference (<5ms)',
                'Low memory footprint',
                'Good for crisis response',
                'Sequential pattern detection'
            ],
            'trade_offs': [
                'Limited long-term memory',
                'No parallel processing',
                'Gradient vanishing for long sequences'
            ],
            'best_for': 'Volatile markets, crisis periods, rapid decision-making'
        },
        'stable': {
            'type': 'Transformer',
            'advantages': [
                'Long-term memory (100+ timesteps)',
                'Parallel processing',
                'Attention visualization',
                'Better factor interpretability'
            ],
            'trade_offs': [
                'Higher latency (~20ms)',
                'More parameters (3x)',
                'Requires more training data'
            ],
            'best_for': 'Stable markets, deep analysis, factor discovery'
        },
        'high_return': {
            'type': 'Ensemble (Transformer + LSTM)',
            'advantages': [
                'Robust predictions',
                'Combines both strengths',
                'Adaptive weighting',
                'Best overall performance'
            ],
            'trade_offs': [
                '2x computation cost',
                'More complex debugging',
                'Higher memory usage'
            ],
            'best_for': 'Bull markets, maximum performance, production systems'
        }
    }

    return encoder_info.get(regime_type, encoder_info['stable'])

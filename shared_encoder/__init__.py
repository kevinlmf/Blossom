"""
Shared Encoder Module (Latent Factor Learner)

Purpose:
- Provide shared market representation for all agents
- Generate learned latent factors z_t = [f₁, f₂, f₃, ...]
- Enable factor & beta estimation for interpretability

Input:
- Global market state
- Agent-specific states from HFT/MFT/LFT

Output:
- Latent representation z_t (endogenous market drivers)
- Factor exposures and loadings

This module is shared across all agents to ensure consistent
market representation and facilitate multi-agent coordination.
"""

from .encoder import SharedEncoder, CNNLSTMEncoder
from .factor_extractor import FactorExtractor
from .encoder_factory import create_encoder, get_encoder_info, EnsembleEncoder
from .em_encoder import (
    EMEncoder,
    ReturnPredictionHead,
    create_em_encoder,
    e_step,
    m_step,
    compute_r_squared
)
from .em_training import EMReturnLearning

__all__ = [
    'SharedEncoder',
    'CNNLSTMEncoder',
    'EnsembleEncoder',
    'FactorExtractor',
    'create_encoder',
    'get_encoder_info',
    # EM算法相关
    'EMEncoder',
    'ReturnPredictionHead',
    'create_em_encoder',
    'e_step',
    'm_step',
    'compute_r_squared',
    'EMReturnLearning'
]

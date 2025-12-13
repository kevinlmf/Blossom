"""
Risk Controller Module

Purpose:
- Copula-based tail dependence modeling
- Monitor joint risk of {HFT, MFT, LFT} agents
- Penalize systemic correlation spikes in allocator reward

Features:
- Gaussian Copula for multivariate risk modeling
- CVaR (Conditional Value at Risk) estimation
- Tail dependence coefficients
- Stress testing and scenario analysis
- Dynamic conditional correlation (DCC)
- Advanced copula methods
"""

from .dynamic_risk_controller import DynamicRiskController
from .cvar_drawdown_controller import (
    CVaRDrawdownController,
    RiskLimits,
    RiskStatus,
    create_risk_controller
)

__all__ = [
    'DynamicRiskController',
    'CVaRDrawdownController',
    'RiskLimits',
    'RiskStatus',
    'create_risk_controller'
]

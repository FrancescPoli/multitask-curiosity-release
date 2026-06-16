"""
Value functions for transforming objective rewards (e.g. learning progress)
into subjective utility, inspired by Prospect Theory.
"""

import math
from typing import Callable

def make_value_function(
    func_type: str,
    scale: float = 0.1,
    loss_weight: float = 2.25
) -> Callable[[float], float]:
    """
    Factory for value functions v(x).
    
    Args:
        func_type: 'identity' or 'tanh_asym'
        scale: sigma parameter (width of linear region)
        loss_weight: lambda parameter (asymmetry for losses)
        
    Returns:
        A callable that maps float -> float.
    """
    if func_type == 'identity':
        return lambda x: float(x)
        
    elif func_type == 'tanh_asym':
        # v(x) = tanh(x/scale)             if x >= 0
        # v(x) = loss_weight * tanh(x/scale) if x < 0
        #
        # Note: input x is typically small (e.g. 0.01 - 0.1).
        # We need to ensure scale is appropriate.
        def v(x: float) -> float:
            if x >= 0:
                return math.tanh(x / scale)
            else:
                return loss_weight * math.tanh(x / scale)
        return v
        
    else:
        raise ValueError(f"Unknown value function type: {func_type}")

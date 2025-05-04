"""
Specialist agent focused on the Bull Put Spread options strategy.
"""
import torch
import torch.nn as nn
import numpy as np
from .base_specialist import BaseSpecialist

class BullPutSpecialist(BaseSpecialist):
    """
    Bull Put Spread specialist agent.
    
    The Bull Put Spread strategy involves:
    - Selling a put at a higher strike price
    - Buying a put at a lower strike price
    
    This creates a bullish position with limited risk and limited profit potential.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(BullPutSpecialist, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, "BULL PUT"
        )
    
    def _initialize_parameters(self):
        """Initialize parameters with Bull Put specific biases"""
        # Set bias for execute head to slightly prefer trading (index 1)
        if hasattr(self.execute_head, 'bias'):
            self.execute_head.bias.data[1] += 0.2
        
        # Set up strike selection biases to prefer proper spacing for bull put spread
        # (typically wants 2 strikes with some distance between them)
        if len(self.strike_heads) >= 2:
            # Bias for choosing strikes with proper width between them
            for i in range(len(self.strike_heads)):
                head = self.strike_heads[i]
                if hasattr(head, 'bias') and head.bias is not None:
                    if i == 0:  # Lower long put (protection)
                        head.bias.data[head.bias.shape[0] // 4] += 0.3  # Bias toward further OTM puts
                    elif i == 1:  # Short put (income)
                        head.bias.data[head.bias.shape[0] * 2 // 5] += 0.4  # Bias toward slightly OTM puts
    
    def get_strategy_specific_features(self, observations):
        """
        Extract Bull Put specific features from observations
        
        Args:
            observations: Observation tensor
            
        Returns:
            Tensor: Bull Put specific features
        """
        batch_size = observations.shape[0]
        features = torch.zeros(batch_size, 10, device=observations.device)
        
        # Extract underlying price
        underlying_price = observations[:, 0] * 1000  # Reverse the normalization
        
        # Market data from observation
        put_prices = observations[:, 101:201].reshape(batch_size, 5, 20)  # 5 exp x 20 strikes
        
        # Compute average put prices as a proxy for put skew and demand
        avg_put_price = torch.mean(put_prices, dim=(1, 2))
        
        # Compare puts across different expirations to estimate term structure
        if put_prices.shape[1] >= 3:
            near_term_puts = torch.mean(put_prices[:, 0], dim=1)
            mid_term_puts = torch.mean(put_prices[:, 2], dim=1)
            term_structure = mid_term_puts / (near_term_puts + 1e-6)
        else:
            term_structure = torch.ones(batch_size, device=observations.device)
        
        # Extract account state
        capital = observations[:, 201]
        portfolio_value = observations[:, 202]
        days_remaining = observations[:, 203]
        
        # Bull Put suitability features:
        # 1. Put price levels (higher prices favor bull put spreads)
        # 2. Term structure (upward sloping term structure favors bull puts)
        # 3. Days to expiration preference (30-60 days ideal for bull puts)
        
        # Populate features tensor
        features[:, 0] = avg_put_price  # Average put price
        features[:, 1] = capital  # Available capital
        features[:, 2] = portfolio_value  # Portfolio value
        features[:, 3] = days_remaining  # Days remaining in episode
        features[:, 4] = term_structure  # Term structure indicator
        
        # Bull Put specific metrics:
        features[:, 5] = torch.clamp(avg_put_price * 5, 0, 1)  # Put premium suitability
        features[:, 6] = torch.clamp((60 - days_remaining) / 60, 0, 1)  # Days to expiration suitability
        features[:, 7] = torch.clamp(term_structure, 0, 2) / 2  # Term structure suitability
        
        return features

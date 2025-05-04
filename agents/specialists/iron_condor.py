"""
Specialist agent focused on the Iron Condor options strategy.
"""
import torch
import torch.nn as nn
import numpy as np
from .base_specialist import BaseSpecialist

class IronCondorSpecialist(BaseSpecialist):
    """
    Iron Condor specialist agent.
    
    The Iron Condor strategy involves:
    - Selling a call spread (short call + long call at higher strike)
    - Selling a put spread (short put + long put at lower strike)
    
    This creates a profit zone between the short strikes with limited risk.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(IronCondorSpecialist, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, "IRON CONDOR"
        )
    
    def _initialize_parameters(self):
        """Initialize parameters with Iron Condor specific biases"""
        # Set bias for execute head to slightly prefer trading (index 1)
        if hasattr(self.execute_head, 'bias'):
            self.execute_head.bias.data[1] += 0.1
        
        # Set up strike selection biases to prefer proper spacing for iron condor
        # (typically wants 4 strikes with appropriate distance between them)
        if len(self.strike_heads) >= 4:
            # Bias for choosing strikes with proper width between them
            # This encourages the agent to select strikes that form a proper iron condor
            # with reasonable width between the short strikes and protection from the long strikes
            for i in range(len(self.strike_heads)):
                head = self.strike_heads[i]
                if hasattr(head, 'bias') and head.bias is not None:
                    if i == 0:  # Lower long put
                        head.bias.data[head.bias.shape[0] // 4] += 0.2  # Bias toward OTM puts
                    elif i == 1:  # Short put
                        head.bias.data[head.bias.shape[0] // 3] += 0.3  # Bias toward slightly OTM puts
                    elif i == 2:  # Short call
                        head.bias.data[head.bias.shape[0] * 2 // 3] += 0.3  # Bias toward slightly OTM calls
                    elif i == 3:  # Upper long call
                        head.bias.data[head.bias.shape[0] * 3 // 4] += 0.2  # Bias toward OTM calls
    
    def get_strategy_specific_features(self, observations):
        """
        Extract Iron Condor specific features from observations
        
        Args:
            observations: Observation tensor
            
        Returns:
            Tensor: Iron Condor specific features
        """
        batch_size = observations.shape[0]
        features = torch.zeros(batch_size, 10, device=observations.device)
        
        # Extract underlying price
        underlying_price = observations[:, 0] * 1000  # Reverse the normalization
        
        # Market volatility estimate (simplified)
        # Use the average of call and put prices as a proxy for implied volatility
        call_prices = observations[:, 1:101].reshape(batch_size, 5, 20)  # 5 exp x 20 strikes
        put_prices = observations[:, 101:201].reshape(batch_size, 5, 20)  # 5 exp x 20 strikes
        
        avg_call_price = torch.mean(call_prices, dim=(1, 2))
        avg_put_price = torch.mean(put_prices, dim=(1, 2))
        implied_vol_estimate = (avg_call_price + avg_put_price) / 2
        
        # Extract account state
        capital = observations[:, 201]
        portfolio_value = observations[:, 202]
        days_remaining = observations[:, 203]
        
        # Iron Condor suitability features:
        # 1. IV percentile (higher IV favors iron condors)
        # 2. Price stability (stable prices favor iron condors)
        # 3. Days to expiration preference (30-45 days ideal for iron condors)
        
        # Populate features tensor
        features[:, 0] = implied_vol_estimate  # Estimated IV
        features[:, 1] = capital  # Available capital
        features[:, 2] = portfolio_value  # Portfolio value
        features[:, 3] = days_remaining  # Days remaining in episode
        features[:, 4] = avg_call_price / avg_put_price  # Call/put ratio (skew indicator)
        
        # Iron Condor specific metrics:
        features[:, 5] = torch.clamp(implied_vol_estimate * 10, 0, 1)  # IV suitability score
        features[:, 6] = torch.clamp((45 - days_remaining) / 45, 0, 1)  # Days to expiration suitability
        
        return features

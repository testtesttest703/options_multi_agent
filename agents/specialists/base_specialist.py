"""
Base class for specialist agents that focus on a single options trading strategy.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, ModelConfigDict, TensorType

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)

class BaseSpecialist(TorchModelV2, nn.Module):
    """
    Base specialist agent implementation using PyTorch.
    Specialists are experts in a specific options trading strategy.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, strategy_name):
        """
        Initialize the specialist model
        
        Args:
            obs_space: Observation space
            action_space: Action space
            num_outputs: Number of output units
            model_config: Model configuration
            name: Model name
            strategy_name: Name of the strategy this specialist focuses on
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.strategy_name = strategy_name
        self.obs_dim = obs_space.shape[0]
        
        # Network dimensions
        hidden_dims = model_config.get("fcnet_hiddens", [256, 128])
        activation = model_config.get("fcnet_activation", "relu")
        
        # Feature extractor with strategy-specific architecture
        self.feature_extractor = self._build_feature_extractor(hidden_dims, activation)
        
        # Action heads
        self.expiry_head = SlimFC(hidden_dims[-1], action_space.spaces['expiry_idx'].n, activation_fn=None)
        
        # Multiple strike selection heads
        self.strike_heads = nn.ModuleList()
        for i in range(len(action_space.spaces['strike_indices'].nvec)):
            self.strike_heads.append(
                SlimFC(hidden_dims[-1], action_space.spaces['strike_indices'].nvec[i], activation_fn=None)
            )
        
        # Execute trade decision head
        self.execute_head = SlimFC(hidden_dims[-1], 2, activation_fn=None)
        
        # Value function head
        self.value_head = SlimFC(hidden_dims[-1], 1, activation_fn=None)
        
        # Initialize parameters with strategy-specific biases
        self._initialize_parameters()
    
    def _build_feature_extractor(self, hidden_dims, activation):
        """Build the feature extraction network"""
        layers = []
        in_size = self.obs_dim
        
        # Create hidden layers
        for out_size in hidden_dims:
            layers.append(SlimFC(in_size, out_size, activation_fn=activation))
            in_size = out_size
        
        return nn.Sequential(*layers)
    
    def _initialize_parameters(self):
        """Initialize parameters with strategy-specific biases"""
        # Default implementation - override in subclasses
        pass
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass of the model
        
        Args:
            input_dict: Dictionary containing inputs
            state: RNN state
            seq_lens: Sequence lengths for RNN
            
        Returns:
            Tensor: Policy outputs
            List: New RNN state
        """
        x = input_dict["obs"].float()
        features = self.feature_extractor(x)
        
        # Store features for value function
        self._features = features
        
        # Return dummy output - we'll use the custom action distribution
        return features, state
    
    def get_action_logits(self, features):
        """
        Get logits for all action components
        
        Args:
            features: Features from feature extractor
            
        Returns:
            dict: Dictionary of logits for each action component
        """
        expiry_logits = self.expiry_head(features)
        
        strike_logits = []
        for head in self.strike_heads:
            strike_logits.append(head(features))
        
        execute_logits = self.execute_head(features)
        
        return {
            'expiry': expiry_logits,
            'strikes': strike_logits,
            'execute': execute_logits
        }
    
    @override(TorchModelV2)
    def value_function(self):
        """
        Get the value function prediction
        
        Returns:
            Tensor: Value function prediction
        """
        return self.value_head(self._features).squeeze(1)
    
    def get_strategy_specific_features(self, observations):
        """
        Extract strategy-specific features from observations
        
        Args:
            observations: Observation tensor
            
        Returns:
            Tensor: Strategy-specific features
        """
        # Default implementation - override in subclasses
        return torch.zeros(observations.shape[0], 10)

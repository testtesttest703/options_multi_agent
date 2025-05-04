"""
Manager agent that coordinates specialist agents for options trading.
The manager evaluates proposals from specialists and selects the best actions.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)

class ManagerAgent(TorchModelV2, nn.Module):
    """
    Manager agent that evaluates and selects from specialist proposals.
    The manager learns to rank proposals and optimize portfolio performance.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        Initialize the manager model
        
        Args:
            obs_space: Observation space
            action_space: Action space
            num_outputs: Number of output units
            model_config: Model configuration
            name: Model name
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.obs_dim = obs_space.shape[0]
        
        # Number of strategies and proposals per strategy
        self.num_strategies = action_space.spaces['strategy_idx'].n
        self.proposals_per_strategy = 5
        
        # Calculate proposal feature size
        # Base features + strategy-specific embeddings + proposal features
        base_feature_size = 4  # underlying price + capital + portfolio_value + days_remaining
        proposal_feature_size = 2  # Basic proposal features (pop, risk_reward)
        total_proposal_features = self.num_strategies * self.proposals_per_strategy * proposal_feature_size
        
        # Network dimensions
        hidden_dims = model_config.get("fcnet_hiddens", [384, 256, 128])
        activation = model_config.get("fcnet_activation", "relu")
        
        # Strategy embeddings (learnable representations for each strategy type)
        self.strategy_embedding = nn.Embedding(self.num_strategies, 8)
        
        # Feature extraction network
        layers = []
        
        # First layer processes base market features
        self.market_encoder = SlimFC(base_feature_size, 64, activation_fn=activation)
        
        # Main feature network
        in_size = 64 + total_proposal_features + (self.num_strategies * 8)  # market + proposals + strategy embeddings
        for out_size in hidden_dims:
            layers.append(SlimFC(in_size, out_size, activation_fn=activation))
            in_size = out_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        self.strategy_head = SlimFC(hidden_dims[-1], self.num_strategies, activation_fn=None)
        self.proposal_head = SlimFC(hidden_dims[-1], self.proposals_per_strategy, activation_fn=None)
        self.execute_head = SlimFC(hidden_dims[-1], 2, activation_fn=None)
        
        # Value function head
        self.value_head = SlimFC(hidden_dims[-1], 1, activation_fn=None)
        
        # Initialize weights to avoid initially strong biases
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters with appropriate biases"""
        # Initialize strategy embedding weights
        nn.init.normal_(self.strategy_embedding.weight, mean=0.0, std=0.1)
        
        # Slight bias toward not executing (conservative default)
        if hasattr(self.execute_head, 'bias'):
            self.execute_head.bias.data[0] += 0.1  # Index 0 = don't execute
    
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
        obs = input_dict["obs"].float()
        
        # Extract base features (first 4 elements)
        base_features = obs[:, :4]  # underlying price, capital, portfolio, days
        
        # Process base market features
        market_features = self.market_encoder(base_features)
        
        # Create strategy embeddings for all strategies
        all_strategy_embeddings = self.strategy_embedding.weight.unsqueeze(0).expand(obs.shape[0], -1, -1)
        all_strategy_embeddings = all_strategy_embeddings.reshape(obs.shape[0], -1)
        
        # Extract proposal features
        proposal_features_start = 4
        proposal_features_end = proposal_features_start + (self.num_strategies * self.proposals_per_strategy * 2)
        proposal_features = obs[:, proposal_features_start:proposal_features_end]
        
        # Combine all features
        combined_features = torch.cat([
            market_features,
            all_strategy_embeddings,
            proposal_features
        ], dim=1)
        
        # Process through main network
        features = self.feature_extractor(combined_features)
        
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
        strategy_logits = self.strategy_head(features)
        proposal_logits = self.proposal_head(features)
        execute_logits = self.execute_head(features)
        
        return {
            'strategy_idx': strategy_logits,
            'proposal_idx': proposal_logits,
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

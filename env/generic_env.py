import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import json
from datetime import datetime
import logging
from collections import defaultdict
import os
import time
import signal
import threading
import io
from functools import wraps

logger = logging.getLogger("OptionsEnv")

def timeout(seconds=10):
    """Timeout decorator for functions that might hang"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

            # Set the timeout handler
            original_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutError as e:
                logger.warning(str(e))
                result = kwargs.get('default', None)
            finally:
                # Reset the alarm and restore original handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, original_handler)
            return result
        return wrapper
    return decorator

class OptionsBaseEnv(gym.Env):
    """
    Generic options environment that can be configured as either:
    1. A specialist environment for a specific strategy
    2. A manager environment that coordinates specialist agents
    """
    def __init__(self, config):
        super(OptionsBaseEnv, self).__init__()

        # Initialize logger at the top level
        import logging
        logger = logging.getLogger("OptionsEnv")

        try:
            self.raw_data = config["data"]
        except KeyError:
            # Data missing, attempt to load from file directly
            import os
            import json
            logger.warning("Data missing in config, loading from file...")

            # Path where main process will save the data
            data_path = os.path.join("/home/ubuntu", "ray_temp", "options_data.json")

            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    self.raw_data = json.load(f)
                logger.info("Successfully loaded data from file")
            else:
                logger.error(f"Data file not found at {data_path}")
                raise ValueError("Cannot initialize environment without data")

        # Define default spaces to prevent the AttributeError
        # These will be replaced by proper spaces in the setup methods
        import numpy as np
        self.action_space = spaces.Dict({
            'expiry_idx': spaces.Discrete(5),
            'strike_indices': spaces.MultiDiscrete([20, 20]),
            'execute': spaces.Discrete(2)
        })
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(200,), dtype=np.float32
        )

        # Continue with the rest of your initialization
        self.config = config
        self.is_specialist = config.get("is_specialist", False)
        self.is_manager = config.get("is_manager", False)
        self.is_evaluation = config.get("is_evaluation", False)
        self.is_recommendation = config.get("is_recommendation", False)
        self.use_latest_data = config.get("use_latest_data", False)
        self.skip_problematic_days = config.get("skip_problematic_days", True)

        # Debug the raw_data
        all_keys = list(self.raw_data.keys())
        if all_keys:
            logger.info(f"Sample keys: {all_keys[:5]}")
        else:
            logger.error("Raw data dictionary is empty!")
            # Create a minimal fake dataset to prevent crashing
            self.raw_data = {"01/01/2025": {"underlyingPrice": 100.0}}
            all_keys = ["01/01/2025"]

        # Set trading days (no filtering for now to diagnose the issue)
        self.trading_days = np.array(sorted(all_keys))[:-1]
        print(self.trading_days)
        logger.info(f"Using {len(self.trading_days)} trading days total")

        if len(self.trading_days) == 0:
            logger.error("No trading days available! Adding default placeholder day to prevent crashes.")
            self.trading_days = ["01/01/2025"]
            self.raw_data["01/01/2025"] = {"underlyingPrice": 100.0}

        # Common parameters
        self.train_days = max(1, int(len(self.trading_days) * 0.7))
        self.current_day_idx = 0
        self.max_day_idx = max(0, min(self.train_days - 1, len(self.trading_days) - 1))
        self.initial_capital = 10000
        self.capital = self.initial_capital
        self.portfolio = []

        # Track the problematic day (day 925)
        self.problematic_day_index = 924  # 0-indexed, so day 925 is at index 924

        # Set up different modes
        try:
            if self.is_specialist:
                if "strategy_name" in config:
                    self._setup_specialist_mode(config["strategy_name"])
                else:
                    logger.warning("Specialist mode specified but no strategy_name provided. Using default")
                    self._setup_specialist_mode("vertical_spread")  # Default strategy
            elif self.is_manager:
                if "strategies" in config:
                    self._setup_manager_mode(config["strategies"], config.get("specialist_paths", {}))
                else:
                    logger.warning("Manager mode specified but no strategies provided. Using defaults")
                    self._setup_manager_mode(["vertical_spread"], {})
            elif self.is_evaluation:
                if "strategies" in config and "specialist_paths" in config:
                    self._setup_evaluation_mode(config["strategies"], config["specialist_paths"])
                else:
                    logger.warning("Evaluation mode specified but missing parameters. Using defaults")
                    self._setup_evaluation_mode(["vertical_spread"], {})
            elif self.is_recommendation:
                if "strategies" in config and "specialist_paths" in config:
                    self._setup_recommendation_mode(config["strategies"], config["specialist_paths"])
                else:
                    logger.warning("Recommendation mode specified but missing parameters. Using defaults")
                    self._setup_recommendation_mode(["vertical_spread"], {})
            else:
                # Default to specialist mode with a default strategy
                logger.warning("No specific mode provided. Defaulting to specialist mode.")
                self._setup_specialist_mode("vertical_spread")
        except Exception as e:
            logger.error(f"Error setting up environment mode: {e}")
            # Don't re-raise - we already have default spaces defined

        # Performance tracking
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        self.returns = []

        # Use the latest data for recommendations
        if self.use_latest_data:
            self.current_day_idx = len(self.trading_days) - 1

    def _setup_specialist_mode(self, strategy_name):
        """Configure environment for specialist agent"""
        self.strategy_name = strategy_name
        from .constants import STRATEGY_UNIQUE_STRIKES, STRATEGY_LEGS

        # Get strategy details
        self.unique_strikes_needed = STRATEGY_UNIQUE_STRIKES.get(strategy_name, 2)
        self.strategy_legs = STRATEGY_LEGS.get(strategy_name, [])

        # Define the specialist observation space
        observation_dim = (
            1 +  # underlying price
            5 * 20 * 2 +  # reduced option data (5 expirations, 20 strikes, prices/IVs)
            3 +  # capital, portfolio value, days remaining
            self.unique_strikes_needed * 5  # simplified position data
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32
        )

        # Define the specialist action space (simpler than before)
        self.action_space = spaces.Dict({
            'expiry_idx': spaces.Discrete(5),  # Select from 5 expirations
            'strike_indices': spaces.MultiDiscrete([20] * self.unique_strikes_needed),  # Select strikes
            'execute': spaces.Discrete(2)  # Execute trade or not
        })

    def _setup_manager_mode(self, strategies, specialist_paths):
        """Configure environment for manager agent"""
        self.strategies = strategies
        self.specialist_paths = specialist_paths
        self.specialists = self._load_specialists()

        # Define manager observation space
        proposal_dim = 10  # Features per strategy proposal
        observation_dim = (
            1 +  # underlying price
            3 +  # capital, portfolio value, days remaining
            len(strategies) * proposal_dim  # Specialist proposal features
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32
        )

        # Define manager action space
        self.action_space = spaces.Dict({
            'strategy_idx': spaces.Discrete(len(strategies)),  # Which strategy to use
            'proposal_idx': spaces.Discrete(5),  # Which proposal from that strategy (top 5)
            'execute': spaces.Discrete(2)  # Execute trade or not
        })

        # Track opportunity cost
        self.last_proposals = []

    def _setup_evaluation_mode(self, strategies, specialist_paths):
        """Configure environment for evaluation"""
        self._setup_manager_mode(strategies, specialist_paths)

        # Use test data
        self.current_day_idx = self.train_days
        self.max_day_idx = len(self.trading_days) - 1

        # Additional tracking for evaluation
        self.trade_history = []
        self.daily_portfolio_values = []

    def _setup_recommendation_mode(self, strategies, specialist_paths):
        """Configure environment for generating recommendations"""
        self._setup_manager_mode(strategies, specialist_paths)

        # Use the most recent data
        self.current_day_idx = len(self.trading_days) - 1
        self.max_day_idx = self.current_day_idx + 1

        # Store recommendations
        self.recommendations = []

    def _load_specialists(self):
        """
        Load specialist agents with deadlock prevention.
        This is the critical method that causes stalling due to circular dependencies.
        """
        specialists = {}
        
        # Skip loading specialists if configured to use mock specialists
        if self.config.get("skip_specialists", False) or self.config.get("use_mock_specialists", False):
            logger.info("Skipping specialist loading (using mock specialists)")
            return specialists
            
        # If deferred loading is enabled, just return empty dict
        if self.config.get("load_specialists_on_demand", False):
            logger.info("Deferring specialist model loading until needed")
            # Store paths for later loading
            self._deferred_specialist_paths = self.specialist_paths
            self._specialists_loaded = False
            return specialists
            
        # Set up deadlock prevention if requested
        if self.config.get("sequential_loading", False) or self.config.get("prevent_deadlock", False):
            logger.info("Using sequential specialist loading to prevent deadlocks")
            # Force single-threaded torch operations temporarily
            try:
                import torch
                original_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                logger.info(f"Set torch to single-threaded mode (was {original_threads} threads)")
            except:
                original_threads = None
                
            # Create a lock to ensure only one model loads at a time
            self._specialist_load_lock = threading.RLock()
        
        try:
            from ray.rllib.algorithms.ppo import PPOConfig
            
            # Load specialists one at a time to prevent deadlocks
            for strategy, path in self.specialist_paths.items():
                if not os.path.exists(path):
                    logger.warning(f"Specialist path not found: {path}")
                    continue
                
                try:
                    logger.info(f"Loading specialist for {strategy} from {path}")
                    
                    # Sequential loading with lock if enabled
                    if self.config.get("sequential_loading", False) or self.config.get("prevent_deadlock", False):
                        with self._specialist_load_lock:
                            # Create proper config for this specialist
                            specialist_env_config = {
                                "data": self.raw_data,
                                "strategy_name": strategy,
                                "is_specialist": True,
                                "ticker": self.config.get("ticker", ""),
                                "skip_problematic_days": True
                            }
                            
                            # Build config and agent
                            config = PPOConfig().environment(
                                OptionsBaseEnv,
                                env_config=specialist_env_config
                            ).framework("torch")
                            
                            agent = config.build()
                            agent.restore(path)
                            specialists[strategy] = agent
                            
                            # Force GC after each specialist is loaded to reduce memory issues
                            import gc
                            gc.collect()
                    else:
                        # Original loading method
                        specialist_env_config = {
                            "data": self.raw_data,
                            "strategy_name": strategy,
                            "is_specialist": True,
                            "ticker": self.config.get("ticker", ""),
                            "skip_problematic_days": True
                        }
                        
                        config = PPOConfig().environment(
                            OptionsBaseEnv,
                            env_config=specialist_env_config
                        ).framework("torch")
                        
                        agent = config.build()
                        agent.restore(path)
                        specialists[strategy] = agent
                        
                except Exception as e:
                    logger.error(f"Error loading specialist for {strategy}: {e}")
                    continue  # Skip this specialist and try the next
        except Exception as e:
            logger.error(f"Error loading specialists: {e}")
        finally:
            # Restore original thread count if we changed it
            if self.config.get("sequential_loading", False) or self.config.get("prevent_deadlock", False):
                if original_threads is not None:
                    import torch
                    torch.set_num_threads(original_threads)
                    logger.info(f"Restored torch thread count to {original_threads}")
                    
        return specialists
        
    def _load_specialist_safely(self, strategy):
        """
        Load a single specialist model with deadlock prevention.
        Used for on-demand loading.
        """
        if not hasattr(self, "_deferred_specialist_paths"):
            logger.warning("No deferred specialist paths available")
            return None
            
        if strategy not in self._deferred_specialist_paths:
            logger.warning(f"No path for specialist strategy: {strategy}")
            return None
            
        path = self._deferred_specialist_paths[strategy]
        
        # Set up deadlock prevention
        import torch
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)  # Force single-threaded for loading
        logger.info(f"Loading specialist {strategy} with single-threaded mode (was {original_threads})")
        
        if not hasattr(self, "_specialist_load_lock"):
            self._specialist_load_lock = threading.RLock()
            
        try:
            with self._specialist_load_lock:  # Ensure sequential loading
                from ray.rllib.algorithms.ppo import PPOConfig
                
                # Create proper config for this specialist
                specialist_env_config = {
                    "data": self.raw_data,
                    "strategy_name": strategy,
                    "is_specialist": True,
                    "ticker": self.config.get("ticker", ""),
                    "skip_problematic_days": True
                }
                
                # Build config and agent
                config = PPOConfig().environment(
                    OptionsBaseEnv,
                    env_config=specialist_env_config
                ).framework("torch")
                
                # Check for actual checkpoint files in the path directory
                import os
                import re
                
                if os.path.isdir(path):
                    checkpoint_files = [f for f in os.listdir(path) if re.match(r"checkpoint-\d+", f)]
                    if checkpoint_files:
                        logger.info(f"Found {len(checkpoint_files)} checkpoints in {path}")
                
                # Clean up memory before loading
                import gc
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Load the agent
                agent = config.build()
                agent.restore(path)
                
                if strategy not in self.specialists:
                    self.specialists[strategy] = agent
                    
                return agent
        except Exception as e:
            logger.error(f"Error in _load_specialist_safely for {strategy}: {e}")
            return None
        finally:
            # Restore original thread count
            torch.set_num_threads(original_threads)

    @timeout(60)
    def step(self, action):
        """Take a step in the environment based on mode"""
        # Check if we're about to process day 925 (index 924) and skip if needed
        if self.skip_problematic_days and self.current_day_idx == self.problematic_day_index:
            logger.warning(f"Skipping problematic day {self.current_day_idx+1} (trading day 925)")
            self.current_day_idx += 1

            # If we've gone beyond max day, return done
            if self.current_day_idx >= self.max_day_idx:
                obs = self._get_specialist_observation() if self.is_specialist else self._get_manager_observation()
                return obs, 0, True, False, {"message": "Reached end of trading days after skipping problematic day"}

        # Now process the step normally
        try:
            if self.is_specialist:
                observation, reward, done, info = self._specialist_step(action)
            elif self.is_manager or self.is_evaluation:
                observation, reward, done, info = self._manager_step(action)
            elif self.is_recommendation:
                observation, reward, done, info = self._recommendation_step(action)
            else:
                raise ValueError("Environment mode not properly configured")

            # Convert to Gymnasium format (add truncated=False)
            return observation, reward, done, False, info
        except Exception as e:
            logger.error(f"Error in step at day {self.current_day_idx+1}: {e}")
            # Return a safe fallback
            obs = self._get_specialist_observation() if self.is_specialist else self._get_manager_observation()
            return obs, 0, False, False, {"error": str(e)}

    def _specialist_step(self, action):
        """Step function for specialist agents"""
        # Track values before action
        prev_portfolio_value = self._calculate_portfolio_value()
        prev_total_value = self.capital + prev_portfolio_value

        # Decode action
        expiry_idx = action['expiry_idx']
        strike_indices = action['strike_indices']
        execute_trade = action['execute']

        # Execute the strategy if requested
        success = False
        message = ""
        if execute_trade == 1:
            success, message = self._execute_specialist_strategy(expiry_idx, strike_indices)

        # Check for expired options
        self._check_expired_options()

        # Move to next trading day
        self.current_day_idx += 1

        # Skip problematic day if needed
        if self.skip_problematic_days and self.current_day_idx == self.problematic_day_index:
            logger.warning(f"Skipping problematic day {self.current_day_idx+1} (trading day 925)")
            self.current_day_idx += 1

        done = self.current_day_idx >= self.max_day_idx

        # Calculate new values and reward
        portfolio_value = self._calculate_portfolio_value()
        total_value = self.capital + portfolio_value

        # Calculate reward components
        profit_loss = total_value - prev_total_value
        normalized_pl = profit_loss / prev_total_value if prev_total_value > 0 else 0

        # Advanced metrics
        pop = self._calculate_probability_of_profit()
        risk_reward = self._calculate_risk_reward_ratio()

        # Combined reward
        reward = normalized_pl * 10 + pop * 2 + 1 / (1 + risk_reward)

        # Penalties and terminal reward
        if execute_trade == 1 and not success:
            reward -= 1

        if done:
            final_return = (total_value - self.initial_capital) / self.initial_capital
            reward += final_return * 5
            self.returns.append(final_return)

        # Get the new observation
        observation = self._get_specialist_observation()

        # Info dictionary
        info = {
            'portfolio_value': portfolio_value,
            'capital': self.capital,
            'total_value': total_value,
            'profit_loss': profit_loss,
            'pop': pop,
            'risk_reward': risk_reward,
            'success': success,
            'message': message,
            'underlying_price': self._get_current_market_data()['underlying_price'],
            'win_rate': self.win_count / max(1, self.win_count + self.loss_count),
            'trades': self.total_trades,
            'avg_pop': sum(p.get('pop', 0) for p in self.portfolio) / max(1, len(self.portfolio)),
            'trading_day': self.current_day_idx,
            'max_days': self.max_day_idx
        }

        return observation, reward, done, info

    def _manager_step(self, action):
        """Step function for manager agent"""
        # Track values before action
        prev_portfolio_value = self._calculate_portfolio_value()
        prev_total_value = self.capital + prev_portfolio_value

        # Get proposals from specialists if needed
        if not self.last_proposals:
            self.last_proposals = self._get_specialist_proposals()

        # Decode action
        strategy_idx = action['strategy_idx']
        proposal_idx = action['proposal_idx']
        execute_trade = action['execute']

        # Execute the selected proposal if requested
        success = False
        message = ""
        if execute_trade == 1 and 0 <= strategy_idx < len(self.strategies) and self.last_proposals:
            strategy = self.strategies[strategy_idx]
            if strategy in self.last_proposals and 0 <= proposal_idx < len(self.last_proposals[strategy]):
                proposal = self.last_proposals[strategy][proposal_idx]
                success, message = self._execute_proposal(proposal)

                # For evaluation, track trade details
                if self.is_evaluation and success:
                    trade_info = {
                        'day': self.trading_days[self.current_day_idx],
                        'strategy': strategy,
                        'proposal': proposal,
                        'entry_value': prev_portfolio_value,
                        'entry_capital': self.capital
                    }
                    self.trade_history.append(trade_info)

        # Check for expired options
        self._check_expired_options()

        # Move to next trading day
        self.current_day_idx += 1

        # Skip problematic day if needed
        if self.skip_problematic_days and self.current_day_idx == self.problematic_day_index:
            logger.warning(f"Skipping problematic day {self.current_day_idx+1} (trading day 925)")
            self.current_day_idx += 1

        done = self.current_day_idx >= self.max_day_idx

        # Update trade history with exit values for evaluation
        if self.is_evaluation and self.trade_history and done:
            for trade in self.trade_history:
                if 'exit_value' not in trade:
                    trade['exit_value'] = self._calculate_portfolio_value()
                    trade['exit_capital'] = self.capital
                    trade['profit'] = (trade['exit_value'] + trade['exit_capital']) - (trade['entry_value'] + trade['entry_capital'])

        # Calculate new values
        portfolio_value = self._calculate_portfolio_value()
        total_value = self.capital + portfolio_value

        # Calculate reward components
        profit_loss = total_value - prev_total_value
        normalized_pl = profit_loss / prev_total_value if prev_total_value > 0 else 0

        # Calculate opportunity cost (reward differential between chosen and best proposal)
        opportunity_cost = 0
        if execute_trade == 1 and self.last_proposals:
            best_reward = -np.inf
            for strat_proposals in self.last_proposals.values():
                for prop in strat_proposals:
                    if prop.get('expected_reward', -np.inf) > best_reward:
                        best_reward = prop.get('expected_reward', -np.inf)

            selected_reward = -np.inf
            if strategy in self.last_proposals and 0 <= proposal_idx < len(self.last_proposals[strategy]):
                selected_reward = self.last_proposals[strategy][proposal_idx].get('expected_reward', -np.inf)

            opportunity_cost = max(0, best_reward - selected_reward)

        # Strategy diversity bonus (encourage using different strategies)
        strategy_counts = defaultdict(int)
        for pos in self.portfolio:
            strategy_counts[pos.get('strategy', '')] += 1

        strategy_diversity = len(strategy_counts) / max(1, len(self.strategies))

        # Combined reward for manager
        reward = (
            normalized_pl * 15 +  # Higher weight on P/L for manager
            strategy_diversity * 3 -  # Bonus for diversification
            opportunity_cost * 5    # Penalty for sub-optimal choices
        )

        # Terminal reward
        if done:
            final_return = (total_value - self.initial_capital) / self.initial_capital
            reward += final_return * 10
            self.returns.append(final_return)

        # Clear proposals for next step
        self.last_proposals = []

        # Save portfolio value for evaluation
        if self.is_evaluation:
            self.daily_portfolio_values.append({
                'day': self.trading_days[min(self.current_day_idx - 1, len(self.trading_days) - 1)],
                'value': total_value
            })

        # Get the new observation
        observation = self._get_manager_observation()

        # Info dictionary
        info = {
            'portfolio_value': portfolio_value,
            'capital': self.capital,
            'total_value': total_value,
            'profit_loss': profit_loss,
            'success': success,
            'message': message,
            'underlying_price': self._get_current_market_data()['underlying_price'],
            'win_rate': self.win_count / max(1, self.win_count + self.loss_count),
            'strategy_diversity': strategy_diversity,
            'opportunity_cost': opportunity_cost,
            'trade_executed': success,
            'trading_day': self.current_day_idx,
            'max_days': self.max_day_idx,
            'trade_info': {
                'strategy': self.strategies[strategy_idx] if 0 <= strategy_idx < len(self.strategies) else "",
                'day': self.trading_days[min(self.current_day_idx - 1, len(self.trading_days) - 1)]
            } if success else {}
        }

        return observation, reward, done, info

    def _recommendation_step(self, action):
        """Step function for recommendation mode"""
        # Similar to manager step but focused on generating recommendations

        # Get proposals from specialists if needed
        if not self.last_proposals:
            self.last_proposals = self._get_specialist_proposals()

        # Decode action
        strategy_idx = action['strategy_idx']
        proposal_idx = action['proposal_idx']
        execute_trade = action['execute']

        # Generate recommendation if requested
        recommendation = None
        if execute_trade == 1 and 0 <= strategy_idx < len(self.strategies) and self.last_proposals:
            strategy = self.strategies[strategy_idx]
            if strategy in self.last_proposals and 0 <= proposal_idx < len(self.last_proposals[strategy]):
                proposal = self.last_proposals[strategy][proposal_idx]

                # Format as recommendation
                recommendation = {
                    'strategy': strategy,
                    'expiration': proposal.get('expiration'),
                    'expected_return': proposal.get('expected_reward', 0),
                    'pop': proposal.get('pop', 0),
                    'risk_reward': proposal.get('risk_reward', 0),
                    'legs': proposal.get('legs', []),
                    'underlying_price': self._get_current_market_data()['underlying_price'],
                    'date': self.trading_days[min(self.current_day_idx, len(self.trading_days) - 1)],
                }
                self.recommendations.append(recommendation)

        # Always done after one step in recommendation mode
        done = True

        # Minimal observation and reward
        observation = self._get_manager_observation()
        reward = 0

        # Info dictionary
        info = {
            'recommendation': recommendation,
            'underlying_price': self._get_current_market_data()['underlying_price'],
            'date': self.trading_days[min(self.current_day_idx, len(self.trading_days) - 1)]
        }

        return observation, reward, done, info

    @timeout(15)
    def _get_specialist_proposals(self, default=None):
        """Get trade proposals from all specialist agents with timeout protection"""
        if default is None:
            default = {}

        proposals = {}
        market_data = self._get_current_market_data()

        # Skip problematic day 925
        if self.current_day_idx == self.problematic_day_index and self.skip_problematic_days:
            logger.warning(f"Skipping specialist proposals for problematic day {self.current_day_idx+1}")
            # Generate fallback proposals to maintain functionality
            return self._generate_fallback_proposals(market_data)

        # If we're using on-demand loading, check if specialists are loaded
        if self.config.get("load_specialists_on_demand", False) and not getattr(self, "_specialists_loaded", True):
            # Try to load needed specialists if not loaded already
            for strategy in self.strategies:
                if strategy not in self.specialists:
                    specialist = self._load_specialist_safely(strategy)
                    if specialist:
                        logger.info(f"Successfully loaded specialist for {strategy} on demand")
            
            self._specialists_loaded = True

        # Check if we should use simplified proposals
        use_simplified = self.config.get("use_simplified_proposals", False)
        
        # Create proposals directly without creating new environments
        for strategy in self.strategies:
            try:
                # Check if this specialist is available
                if strategy not in self.specialists:
                    if not use_simplified:
                        logger.warning(f"Specialist for {strategy} not available, using simplified proposals")
                    strategy_proposals = self._generate_simplified_proposals(market_data, strategy)
                    proposals[strategy] = strategy_proposals
                    continue
                    
                # For non-problematic days, try the original approach with protection
                specialist = self.specialists[strategy]
                
                # Configure a temporary environment
                temp_env_config = {
                    "data": self.raw_data,
                    "strategy_name": strategy,
                    "is_specialist": True,
                    "skip_problematic_days": True  # Always skip problematic days
                }

                # Use a placeholder environment to get proper observation
                temp_env = OptionsBaseEnv(temp_env_config)
                temp_env.current_day_idx = self.current_day_idx

                # Get observation from this environment
                observation = temp_env._get_specialist_observation()

                # Sample multiple actions for diverse proposals
                strategy_proposals = []

                # Only try model inference a few times with timeout protection
                for _ in range(3):  # Get 3 proposals (reduced from 5)
                    try:
                        # Use the correct API based on Ray version with a timeout
                        if hasattr(specialist, 'get_module'):
                            # Ray 2.0+ API
                            module = specialist.get_module("default_policy")

                            # Convert observation to tensor
                            import torch
                            obs_tensor = torch.FloatTensor(observation)
                            if len(obs_tensor.shape) == 1:
                                obs_tensor = obs_tensor.unsqueeze(0)

                            # Get action using forward_inference with timeout protection
                            with torch.no_grad():
                                action_dict = module.forward_inference({'obs': obs_tensor})
                                action = action_dict['actions'].squeeze().cpu().numpy()
                        else:
                            # Ray 1.x API with timeout protection
                            action = specialist.compute_single_action(observation)

                        # Execute in temporary environment to evaluate the proposal
                        proposal_obs, proposal_reward, done, _, proposal_info = temp_env.step(action)

                        # Only include executable proposals
                        if action['execute'] == 1 and proposal_info.get('success', False):
                            # Extract strikes based on action
                            expiry_idx = action['expiry_idx']
                            strike_indices = action['strike_indices']

                            # Get available expirations for this day
                            expirations = sorted(market_data['expirations'])
                            if expiry_idx < len(expirations):
                                expiration = expirations[expiry_idx]

                                # Extract legs information
                                legs = self._extract_leg_info(
                                    market_data, strategy, action, expiration
                                )

                                if legs:
                                    proposal = {
                                        'expiration': expiration,
                                        'pop': proposal_info.get('pop', 0),
                                        'risk_reward': proposal_info.get('risk_reward', 0),
                                        'expected_reward': proposal_reward,
                                        'legs': legs,
                                        'raw_action': action
                                    }
                                    strategy_proposals.append(proposal)

                    except Exception as e:
                        logger.error(f"Error getting action for {strategy}: {e}")
                        continue

                # If normal approach yielded no proposals, use simplified approach
                if not strategy_proposals:
                    strategy_proposals = self._generate_simplified_proposals(market_data, strategy)

                # Sort proposals by expected reward and add to results
                if strategy_proposals:
                    strategy_proposals.sort(key=lambda x: x['expected_reward'], reverse=True)
                    proposals[strategy] = strategy_proposals

            except TimeoutError:
                logger.warning(f"Timeout getting proposals for {strategy} - falling back to simplified")
                strategy_proposals = self._generate_simplified_proposals(market_data, strategy)
                if strategy_proposals:
                    proposals[strategy] = strategy_proposals
            except Exception as e:
                logger.error(f"Error getting proposals for {strategy}: {e}")
                # Try simplified approach as fallback
                try:
                    strategy_proposals = self._generate_simplified_proposals(market_data, strategy)
                    if strategy_proposals:
                        proposals[strategy] = strategy_proposals
                except:
                    pass

        # If we couldn't generate any proposals, use fallback
        if not proposals:
            logger.warning("No proposals generated, using fallback proposals")
            return self._generate_fallback_proposals(market_data)

        return proposals

    def _generate_simplified_proposals(self, market_data, strategy):
        """Generate simplified proposals without model inference"""
        from .constants import STRATEGY_LEGS

        strategy_proposals = []
        strategy_legs = STRATEGY_LEGS.get(strategy, [])

        # Get available expirations sorted by date
        expirations = sorted(market_data['expirations'])
        if not expirations:
            return []

        # Generate 5 different proposals with varying expirations/strikes
        for i in range(5):
            try:
                # Choose different expirations for diversity
                exp_idx = min(i % len(expirations), len(expirations)-1)
                expiration = expirations[exp_idx]

                # Get strikes for this expiration
                call_strikes = market_data['call_options'].get(expiration, {}).get('strikes', [])
                put_strikes = market_data['put_options'].get(expiration, {}).get('strikes', [])

                if not call_strikes or not put_strikes:
                    continue

                # Create the legs for this proposal
                legs = []
                for j, leg_template in enumerate(strategy_legs):
                    leg_type = leg_template['type'].lower()
                    position_type = leg_template['position'].split(' ')[0].lower()

                    # Select different strikes for diversity
                    strikes = call_strikes if leg_type == 'call' else put_strikes
                    if not strikes:
                        continue

                    # Choose strike based on moneyness - different for each proposal
                    strike_idx = min(len(strikes)-1, max(0, len(strikes)//2 + (i-2)))
                    if j % 2 == 1:  # For second leg of each pair, offset a bit
                        strike_idx = min(len(strikes)-1, strike_idx + 1 + i % 3)

                    strike = strikes[strike_idx]

                    # Get option price
                    price_array = market_data['call_options' if leg_type == 'call' else 'put_options']\
                                          .get(expiration, {})\
                                          .get('prices', [])

                    if strike_idx < len(price_array):
                        price = price_array[strike_idx]
                    else:
                        price = 1.0  # Default price if unavailable

                    legs.append({
                        'type': leg_type,
                        'position': position_type,
                        'strike': strike,
                        'price': price
                    })

                if legs:
                    # Calculate some reasonable values for the proposal
                    pop = 0.6 - (i * 0.05)  # 0.6, 0.55, 0.5, 0.45, 0.4
                    risk_reward = 1.0 + (i * 0.2)  # 1.0, 1.2, 1.4, 1.6, 1.8
                    expected_reward = 0.8 - (i * 0.1)  # 0.8, 0.7, 0.6, 0.5, 0.4

                    proposal = {
                        'expiration': expiration,
                        'pop': pop,
                        'risk_reward': risk_reward,
                        'expected_reward': expected_reward,
                        'legs': legs
                    }

                    strategy_proposals.append(proposal)

            except Exception as e:
                logger.error(f"Error generating simplified proposal for {strategy}: {e}")
                continue

        return strategy_proposals

    def _generate_fallback_proposals(self, market_data):
        """Generate fallback proposals when all else fails"""
        fallback_proposals = {}

        for strategy in self.strategies:
            proposals = []
            # Create just one simple proposal per strategy
            try:
                expirations = sorted(market_data['expirations'])
                if expirations:
                    expiration = expirations[0]

                    # Simple proposal for any strategy
                    proposals.append({
                        'expiration': expiration,
                        'pop': 0.55,
                        'risk_reward': 1.2,
                        'expected_reward': 0.5,
                        'legs': []  # Empty legs but still valid structure
                    })

                    fallback_proposals[strategy] = proposals
            except:
                # If even this fails, add an empty proposal
                proposals.append({
                    'expiration': "01/01/2025",
                    'pop': 0.5,
                    'risk_reward': 1.0,
                    'expected_reward': 0.4,
                    'legs': []
                })
                fallback_proposals[strategy] = proposals

        return fallback_proposals

    def _extract_leg_info(self, market_data, strategy, action, expiration):
        """Extract detailed leg information for a proposal"""
        from .constants import STRATEGY_LEGS

        legs = []
        strategy_legs = STRATEGY_LEGS.get(strategy, [])
        strike_indices = action['strike_indices']

        for i, leg in enumerate(strategy_legs):
            if i >= len(strike_indices):
                break

            leg_type = leg['type'].lower()
            position_type = leg['position'].split(' ')[0].lower()  # Long/Short
            strike_idx = strike_indices[i]

            # Get strikes for this expiration
            if leg_type == 'call' and expiration in market_data['call_options']:
                strikes = market_data['call_options'][expiration]['strikes']
                prices = market_data['call_options'][expiration]['prices']

                if 0 <= strike_idx < len(strikes):
                    strike = strikes[strike_idx]
                    price = prices[strike_idx]

                    legs.append({
                        'type': 'call',
                        'position': 'long' if position_type == 'long' else 'short',
                        'strike': strike,
                        'price': price
                    })

            elif leg_type == 'put' and expiration in market_data['put_options']:
                strikes = market_data['put_options'][expiration]['strikes']
                prices = market_data['put_options'][expiration]['prices']

                if 0 <= strike_idx < len(strikes):
                    strike = strikes[strike_idx]
                    price = prices[strike_idx]

                    legs.append({
                        'type': 'put',
                        'position': 'long' if position_type == 'long' else 'short',
                        'strike': strike,
                        'price': price
                    })

        return legs

    def _execute_specialist_strategy(self, expiry_idx, strike_indices):
        """Execute a strategy as a specialist agent"""
        market_data = self._get_current_market_data()

        # Get the selected expiration
        expirations = sorted(market_data['expirations'])
        if expiry_idx >= len(expirations):
            return False, "Invalid expiration index"

        expiration = expirations[expiry_idx]

        # Check available strikes
        call_strikes = market_data['call_options'].get(expiration, {}).get('strikes', [])
        put_strikes = market_data['put_options'].get(expiration, {}).get('strikes', [])

        if not call_strikes or not put_strikes:
            return False, f"No options available for expiration {expiration}"

        # Map strike indices to actual strikes
        selected_strikes = []
        for i in range(self.unique_strikes_needed):
            strike_idx = strike_indices[i]
            leg_type = self.strategy_legs[i % len(self.strategy_legs)]['type'].lower()

            if leg_type == 'call' and strike_idx < len(call_strikes):
                selected_strikes.append(call_strikes[strike_idx])
            elif leg_type == 'put' and strike_idx < len(put_strikes):
                selected_strikes.append(put_strikes[strike_idx])
            else:
                # Use first available strike as fallback
                if leg_type == 'call' and call_strikes:
                    selected_strikes.append(call_strikes[0])
                elif leg_type == 'put' and put_strikes:
                    selected_strikes.append(put_strikes[0])
                else:
                    return False, f"No {leg_type} strikes available"

        # Calculate cost and create positions
        total_cost = 0
        new_positions = []

        for i, leg in enumerate(self.strategy_legs):
            if i >= self.unique_strikes_needed:
                break

            leg_type = leg['type'].lower()
            position_type = leg['position'].split(' ')[0].lower()  # Long/Short

            # Get quantity (simplified - 1 contract per leg)
            quantity = 1

            strike_price = selected_strikes[i % len(selected_strikes)]

            # Get option price
            if leg_type == 'call':
                strike_idx = call_strikes.index(strike_price) if strike_price in call_strikes else -1
                if strike_idx == -1:
                    continue  # Skip if strike not found
                option_price = market_data['call_options'][expiration]['prices'][strike_idx]
            else:  # put
                strike_idx = put_strikes.index(strike_price) if strike_price in put_strikes else -1
                if strike_idx == -1:
                    continue  # Skip if strike not found
                option_price = market_data['put_options'][expiration]['prices'][strike_idx]

            # Calculate cost
            position_cost = option_price * quantity * 100  # 100 shares per contract

            if position_type == 'long':
                total_cost += position_cost
                final_quantity = quantity
            else:  # short
                total_cost -= position_cost
                final_quantity = -quantity

            # Create position record
            position = {
                'type': leg_type,
                'position_type': position_type,
                'strike': strike_price,
                'expiration': expiration,
                'quantity': final_quantity,
                'cost_basis': position_cost,
                'current_value': position_cost,
                'entry_day_idx': self.current_day_idx,
                'strategy': self.strategy_name,
                'pop': self._estimate_pop(leg_type, position_type, strike_price, market_data['underlying_price'])
            }

            new_positions.append(position)

        # Check if we have enough capital
        if self.capital < total_cost:
            return False, "Insufficient capital"

        # Execute the trade
        self.capital -= total_cost
        self.portfolio.extend(new_positions)
        self.total_trades += 1

        return True, f"Executed {self.strategy_name} for ${total_cost:.2f}"

    def _execute_proposal(self, proposal):
        """Execute a proposal from a specialist"""
        total_cost = 0
        new_positions = []

        for leg in proposal.get('legs', []):
            leg_type = leg.get('type').lower()
            position_type = leg.get('position').lower()
            strike_price = leg.get('strike')
            price = leg.get('price', 0)
            expiration = proposal.get('expiration')

            # Skip if missing critical info
            if not all([leg_type, position_type, strike_price, expiration]):
                continue

            # Calculate cost
            quantity = 1  # Simplified - 1 contract per leg
            position_cost = price * quantity * 100  # 100 shares per contract

            if position_type == 'long':
                total_cost += position_cost
                final_quantity = quantity
            else:  # short
                total_cost -= position_cost
                final_quantity = -quantity

            # Create position record
            position = {
                'type': leg_type,
                'position_type': position_type,
                'strike': strike_price,
                'expiration': expiration,
                'quantity': final_quantity,
                'cost_basis': position_cost,
                'current_value': position_cost,
                'entry_day_idx': self.current_day_idx,
                'strategy': proposal.get('strategy', ''),
                'pop': leg.get('pop', 0) or proposal.get('pop', 0)
            }

            new_positions.append(position)

        # Check if we have enough capital
        if not new_positions:
            return False, "No valid legs in proposal"

        if self.capital < total_cost:
            return False, "Insufficient capital"

        # Execute the trade
        self.capital -= total_cost
        self.portfolio.extend(new_positions)
        self.total_trades += 1

        return True, f"Executed proposal for ${total_cost:.2f}"

    def _estimate_pop(self, option_type, position_type, strike_price, underlying_price):
        """Estimate probability of profit based on moneyness"""
        moneyness = strike_price / underlying_price

        if option_type == 'call':
            if position_type == 'long':
                # Long call: higher POP when strike is below underlying (ITM)
                pop = max(0.1, min(0.9, 0.5 - (moneyness - 1) * 2))
            else:
                # Short call: higher POP when strike is above underlying (OTM)
                pop = max(0.1, min(0.9, 0.5 + (moneyness - 1) * 2))
        else:  # put
            if position_type == 'long':
                # Long put: higher POP when strike is above underlying (ITM)
                pop = max(0.1, min(0.9, 0.5 + (moneyness - 1) * 2))
            else:
                # Short put: higher POP when strike is below underlying (OTM)
                pop = max(0.1, min(0.9, 0.5 - (moneyness - 1) * 2))

        return pop

    def _check_expired_options(self):
        """Check for and handle expired options"""
        if not self.portfolio:
            return

        try:
            current_day = self.trading_days[min(self.current_day_idx, len(self.trading_days) - 1)]
            market_data = self._get_current_market_data()
            underlying_price = market_data['underlying_price']
            positions_to_remove = []

            for i, position in enumerate(self.portfolio):
                expiration = position['expiration']

                # Check if option has expired (simplified)
                try:
                    print(f"stats: {expiration}, {current_day}")
                    if expiration <= int(time.mktime(time.strptime(f"{current_day}", "%m/%d/%Y"))):  # In real impl, would check actual expiration
                        option_type = position['type'].lower()
                        strike_price = position['strike']
                        quantity = position['quantity']

                        # Calculate settlement value
                        if option_type == 'call':
                            settlement = max(0, underlying_price - strike_price)
                        else:  # put
                            settlement = max(0, strike_price - underlying_price)

                        # Update capital
                        if quantity > 0:  # Long position
                            self.capital += settlement * quantity * 100
                        else:  # Short position
                            self.capital -= settlement * abs(quantity) * 100

                        positions_to_remove.append(i)

                        # Track trade outcome
                        if settlement > position['cost_basis'] / (abs(quantity) * 100):
                            self.win_count += 1
                        else:
                            self.loss_count += 1
                except Exception as e:
                    logger.error(f"Error processing expiration: {e}")
                    # Continue with next position

            # Remove expired positions
            for i in sorted(positions_to_remove, reverse=True):
                self.portfolio.pop(i)
        except Exception as e:
            logger.error(f"Error in _check_expired_options: {e}")
            # Continue execution despite errors

    @timeout(10)
    def _calculate_portfolio_value(self):
        """Calculate the current value of the portfolio"""
        if not self.portfolio:
            return 0

        try:
            market_data = self._get_current_market_data()
            underlying_price = market_data['underlying_price']

            total_value = 0
            for position in self.portfolio:
                option_type = position['type'].lower()
                strike_price = position['strike']
                expiration = position['expiration']
                quantity = position['quantity']

                # Look up current price
                if option_type == 'call' and expiration in market_data['call_options']:
                    strikes = market_data['call_options'][expiration]['strikes']
                    prices = market_data['call_options'][expiration]['prices']

                    if strike_price in strikes:
                        idx = strikes.index(strike_price)
                        current_price = prices[idx]
                    else:
                        closest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - strike_price))
                        current_price = prices[closest_idx]

                elif option_type == 'put' and expiration in market_data['put_options']:
                    strikes = market_data['put_options'][expiration]['strikes']
                    prices = market_data['put_options'][expiration]['prices']

                    if strike_price in strikes:
                        idx = strikes.index(strike_price)
                        current_price = prices[idx]
                    else:
                        closest_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - strike_price))
                        current_price = prices[closest_idx]
                else:
                    # Intrinsic value as fallback
                    if option_type == 'call':
                        current_price = max(0, underlying_price - strike_price)
                    else:  # put
                        current_price = max(0, strike_price - underlying_price)

                position_value = current_price * abs(quantity) * 100  # 100 shares per contract

                if quantity > 0:  # Long position
                    total_value += position_value
                else:  # Short position
                    total_value -= position_value

                # Update position with current value
                position['current_value'] = position_value

            return total_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0  # Return 0 as fallback

    def _calculate_probability_of_profit(self):
        """Estimate probability of profit for current positions"""
        if not self.portfolio:
            return 0.5

        try:
            market_data = self._get_current_market_data()
            underlying_price = market_data['underlying_price']

            # Get POP values from positions or calculate
            pop_values = []
            position_values = []

            for position in self.portfolio:
                # Use pre-calculated POP if available
                if 'pop' in position:
                    pop = position['pop']
                else:
                    # Estimate based on moneyness
                    option_type = position['type'].lower()
                    position_type = position['position_type'].lower()
                    strike_price = position['strike']
                    pop = self._estimate_pop(option_type, position_type, strike_price, underlying_price)

                position_value = abs(position['current_value'])
                position_values.append(position_value)
                pop_values.append(pop)

            # Calculate weighted average POP
            total_value = sum(position_values)
            if total_value == 0:
                return np.mean(pop_values) if pop_values else 0.5

            weighted_pop = sum(pop * val / total_value for pop, val in zip(pop_values, position_values))
            return min(max(weighted_pop, 0), 1)  # Ensure between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating probability of profit: {e}")
            return 0.5  # Return neutral value as fallback

    def _calculate_risk_reward_ratio(self):
        """Calculate risk/reward ratio for portfolio"""
        if not self.portfolio:
            return 1.0

        try:
            # Calculate max profit and loss (simplified)
            max_loss = 0
            max_profit = 0

            for position in self.portfolio:
                option_type = position['type'].lower()
                position_type = position['position_type'].lower()
                strike_price = position['strike']
                quantity = abs(position['quantity'])
                cost_basis = position['cost_basis']

                if position_type == 'long':
                    max_loss += cost_basis
                    if option_type == 'call':
                        max_profit += cost_basis * 3  # Approximation
                    else:  # put
                        max_profit += strike_price * quantity * 100 - cost_basis
                else:  # short
                    max_profit += cost_basis
                    if option_type == 'call':
                        max_loss += cost_basis * 3  # Approximation
                    else:  # put
                        max_loss += strike_price * quantity * 100 - cost_basis

            # Avoid division by zero
            if max_profit <= 0:
                return 5.0  # Default high ratio

            return max_loss / max_profit
        except Exception as e:
            logger.error(f"Error calculating risk reward ratio: {e}")
            return 1.0  # Return neutral value as fallback

    @timeout(10)
    def _get_current_market_data(self):
        """Get preprocessed market data for current trading day with timeout protection"""
        # Avoid accessing problematic days
        if self.current_day_idx == self.problematic_day_index and self.skip_problematic_days:
            logger.warning(f"Attempted to access market data for problematic day {self.current_day_idx+1}")
            # Use a neighboring day's data instead
            safe_idx = max(0, min(self.current_day_idx - 1, len(self.trading_days) - 1))
            trading_day = self.trading_days[safe_idx]
        else:
            # Make sure we don't go out of bounds
            safe_idx = max(0, min(self.current_day_idx, len(self.trading_days) - 1))
            trading_day = self.trading_days[safe_idx]

        try:
            print(f'trading day index: {safe_idx}')
            from utils.data_processor import extract_options_data
            return extract_options_data(self.raw_data, trading_day)
        except Exception as e:
            logger.error(f"Error getting market data for day {trading_day}: {e}")
            # Return minimal default data to prevent crashes
            return {
                'underlying_price': 100.0,
                'expirations': [],
                'call_options': {},
                'put_options': {}
            }

    def _get_specialist_observation(self):
        """Create specialist observation vector"""
        try:
            market_data = self._get_current_market_data()
            underlying_price = market_data['underlying_price']

            # Extract options data (simplified)
            call_prices = np.zeros((5, 20))  # 5 expirations, 20 strikes
            put_prices = np.zeros((5, 20))

            expirations = sorted(market_data['expirations'])[:5]
            for i, exp in enumerate(expirations):
                if i >= 5:
                    break

                if exp in market_data['call_options']:
                    call_data = market_data['call_options'][exp]
                    n_strikes = min(len(call_data['strikes']), 20)
                    if n_strikes > 0:
                        call_prices[i, :n_strikes] = call_data['prices'][:n_strikes]

                if exp in market_data['put_options']:
                    put_data = market_data['put_options'][exp]
                    n_strikes = min(len(put_data['strikes']), 20)
                    if n_strikes > 0:
                        put_prices[i, :n_strikes] = put_data['prices'][:n_strikes]

            # Flatten option data
            # Flatten option data
            option_data = np.concatenate([
                call_prices.flatten(),
                put_prices.flatten()
            ])

            # Portfolio state
            portfolio_value = self._calculate_portfolio_value()
            days_remaining = self.max_day_idx - self.current_day_idx
            account_state = [self.capital, portfolio_value, days_remaining]

            # Position data (simplified)
            position_data = []
            for i in range(self.unique_strikes_needed):
                if i < len(self.portfolio):
                    pos = self.portfolio[i]
                    option_type = 1 if pos['type'].lower() == 'put' else 0
                    strike = pos['strike'] / underlying_price  # Normalize
                    position_data.extend([
                        option_type,
                        strike,
                        pos['quantity'],
                        pos['current_value'] / self.initial_capital,  # Normalize
                        pos['cost_basis'] / self.initial_capital,  # Normalize
                    ])
                else:
                    position_data.extend([0, 0, 0, 0, 0])

            # Combine all features
            observation = np.concatenate([
                [underlying_price / 1000],  # Scale down
                option_data,
                account_state,
                position_data
            ]).astype(np.float32)

            return observation
        except Exception as e:
            logger.error(f"Error creating specialist observation: {e}")
            # Return zeros array of correct shape as fallback
            return np.zeros(self.observation_space.shape).astype(np.float32)

    def _get_manager_observation(self):
        """Create manager observation vector"""
        try:
            market_data = self._get_current_market_data()
            underlying_price = market_data['underlying_price']

            # Get proposals if needed
            if not self.last_proposals:
                self.last_proposals = self._get_specialist_proposals()

            # Portfolio state
            portfolio_value = self._calculate_portfolio_value()
            days_remaining = self.max_day_idx - self.current_day_idx
            account_state = [self.capital / self.initial_capital, portfolio_value / self.initial_capital, days_remaining / 30]

            # Process proposals into fixed-length feature vector
            proposal_features = []

            # For each strategy
            for strategy in self.strategies:
                strategy_props = self.last_proposals.get(strategy, [])

                # For each of the top 5 proposals
                for i in range(5):
                    if i < len(strategy_props):
                        prop = strategy_props[i]
                        # Features: pop, risk/reward, expected reward
                        proposal_features.extend([
                            prop.get('pop', 0),
                            1.0 / (1.0 + prop.get('risk_reward', 1.0)),  # Normalize risk/reward
                        ])
                    else:
                        # Padding for missing proposals
                        proposal_features.extend([0, 0])

            # Pad proposal features if needed
            while len(proposal_features) < len(self.strategies) * 10:  # 10 features per strategy
                proposal_features.append(0)

            # Combine all features
            observation = np.concatenate([
                [underlying_price / 1000],  # Scale down
                account_state,
                proposal_features
            ]).astype(np.float32)

            return observation
        except Exception as e:
            logger.error(f"Error creating manager observation: {e}")
            # Return zeros array of correct shape as fallback
            return np.zeros(self.observation_space.shape).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """Reset the environment - Gymnasium API requires seed and options parameters"""
        # Initialize random number generator with seed
        if seed is not None:
            np.random.seed(seed)

        if not self.is_recommendation:
            # Start from the beginning for training
            if not self.is_evaluation:
                self.current_day_idx = 0

            # Test mode starts after training data
            else:
                self.current_day_idx = self.train_days
        else:
            # Recommendation mode uses latest data
            self.current_day_idx = len(self.trading_days) - 1

        self.capital = self.initial_capital
        self.portfolio = []
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        self.returns = []
        self.last_proposals = []

        if self.is_evaluation:
            self.trade_history = []
            self.daily_portfolio_values = []

        if self.is_recommendation:
            self.recommendations = []

        # Get the observation
        if self.is_specialist:
            observation = self._get_specialist_observation()
        else:
            observation = self._get_manager_observation()

        # Return observation and empty info dict (Gymnasium API)
        return observation, {}

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import json
from datetime import datetime
import logging
from collections import defaultdict
import os

logger = logging.getLogger("OptionsEnv")

class OptionsBaseEnv(gym.Env):
    """
    Generic options environment that can be configured as either:
    1. A specialist environment for a specific strategy
    2. A manager environment that coordinates specialist agents
    """

    def __init__(self, config):
        super(OptionsBaseEnv, self).__init__()

        self.config = config
        self.raw_data = config["data"]
        self.is_specialist = config.get("is_specialist", False)
        self.is_manager = config.get("is_manager", False)
        self.is_evaluation = config.get("is_evaluation", False)
        self.is_recommendation = config.get("is_recommendation", False)
        self.use_latest_data = config.get("use_latest_data", False)

        # Common parameters
        self.trading_days = sorted(list(self.raw_data.keys()))
        self.train_days = int(len(self.trading_days) * 0.7)
        self.current_day_idx = 0
        self.max_day_idx = self.train_days - 1
        self.initial_capital = 10000
        self.capital = self.initial_capital
        self.portfolio = []

        # Set up different modes
        if self.is_specialist:
            self._setup_specialist_mode(config["strategy_name"])
        elif self.is_manager:
            self._setup_manager_mode(config["strategies"], config.get("specialist_paths", {}))
        elif self.is_evaluation:
            self._setup_evaluation_mode(config["strategies"], config["specialist_paths"])
        elif self.is_recommendation:
            self._setup_recommendation_mode(config["strategies"], config["specialist_paths"])

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
        """Load all specialist agents from checkpoints"""
        specialists = {}

        try:
            from ray.rllib.algorithms.ppo import PPOConfig

            for strategy, path in self.specialist_paths.items():
                try:
                    if os.path.exists(path):
                        config = PPOConfig().environment(OptionsBaseEnv).framework("torch")
                        agent = config.build()
                        agent.restore(path)
                        specialists[strategy] = agent
                        logger.info(f"Loaded specialist for {strategy}")
                    else:
                        logger.warning(f"Specialist checkpoint not found for {strategy}: {path}")
                except Exception as e:
                    logger.error(f"Failed to load specialist for {strategy}: {e}")
        except ImportError:
            logger.warning("Ray not available. Skipping specialist loading.")

        return specialists

    def step(self, action):
        """Take a step in the environment based on mode"""
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
            'avg_pop': sum(p.get('pop', 0) for p in self.portfolio) / max(1, len(self.portfolio))
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
                'day': self.trading_days[self.current_day_idx - 1],
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
            'trade_info': {
                'strategy': self.strategies[strategy_idx] if 0 <= strategy_idx < len(self.strategies) else "",
                'day': self.trading_days[self.current_day_idx - 1]
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
                    'date': self.trading_days[self.current_day_idx],
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
            'date': self.trading_days[self.current_day_idx]
        }

        return observation, reward, done, info

    def _get_specialist_proposals(self):
        """Get trade proposals from all specialist agents"""
        proposals = {}

        market_data = self._get_current_market_data()

        # Create a specialist-style observation
        for strategy, specialist in self.specialists.items():
            try:
                # Configure a temporary environment
                temp_env_config = {
                    "data": self.raw_data,
                    "strategy_name": strategy,
                    "is_specialist": True
                }

                # Use a placeholder environment to get proper observation
                temp_env = OptionsBaseEnv(temp_env_config)
                temp_env.current_day_idx = self.current_day_idx

                # Get observation from this environment
                observation = temp_env._get_specialist_observation()

                # Get proposals (actions) from specialist
                strategy_proposals = []

                # Sample multiple actions to get diverse proposals
                for _ in range(5):  # Get 5 proposals
                    action = specialist.compute_single_action(observation)

                    # Execute in temporary environment to evaluate the proposal
                    # Update for Gymnasium compatibility
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

                # Sort proposals by expected reward
                if strategy_proposals:
                    strategy_proposals.sort(key=lambda x: x['expected_reward'], reverse=True)
                    proposals[strategy] = strategy_proposals

            except Exception as e:
                logger.error(f"Error getting proposals for {strategy}: {e}")

        return proposals

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

        current_day = self.trading_days[self.current_day_idx]
        market_data = self._get_current_market_data()
        underlying_price = market_data['underlying_price']
        positions_to_remove = []

        for i, position in enumerate(self.portfolio):
            expiration = position['expiration']

            # Check if option has expired (simplified)
            if expiration <= current_day:  # In real impl, would check actual expiration
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

        # Remove expired positions
        for i in sorted(positions_to_remove, reverse=True):
            self.portfolio.pop(i)

    def _calculate_portfolio_value(self):
        """Calculate the current value of the portfolio"""
        if not self.portfolio:
            return 0

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

    def _calculate_probability_of_profit(self):
        """Estimate probability of profit for current positions"""
        if not self.portfolio:
            return 0.5

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

    def _calculate_risk_reward_ratio(self):
        """Calculate risk/reward ratio for portfolio"""
        if not self.portfolio:
            return 1.0

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

    def _get_current_market_data(self):
        """Get preprocessed market data for current trading day"""
        trading_day = self.trading_days[self.current_day_idx]
        from utils.data_processor import extract_options_data
        return extract_options_data(self.raw_data, trading_day)

    def _get_specialist_observation(self):
        """Create specialist observation vector"""
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

    def _get_manager_observation(self):
        """Create manager observation vector"""
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

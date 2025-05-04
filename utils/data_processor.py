#!/usr/bin/env python3
"""
Multi-Agent Options Trading - Main Entry Point
This script uses Ray RLlib to train specialist agents for options trading strategies
"""
import os
import json
import argparse
import logging
import torch
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from datetime import datetime

# Import our custom modules
from env.generic_env import OptionsBaseEnv
from agents.manager import ManagerAgent
from agents.registry import SPECIALIST_REGISTRY
from utils.data_loader import load_options_data
from utils.metrics import calculate_sharpe_ratio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("options_multi_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OptionsMultiAgent")

def main():
    """Main entry point for the options multi-agent trading system"""
    parser = argparse.ArgumentParser(description="Multi-Agent Options Trading with Ray/RLlib")
    parser.add_argument("--data_file", type=str, required=True, help="Path to options data JSON file")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory for saving models and results")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--strategies", type=str, nargs='+', default=None, help="Specific strategies to train")
    parser.add_argument("--cpu_per_worker", type=int, default=1, help="CPUs per worker")
    parser.add_argument("--num_gpus", type=float, default=0, help="GPUs to use for training (can be fractional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(num_cpus=os.cpu_count(), num_gpus=args.num_gpus)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load the data
    logger.info(f"Loading options data from {args.data_file}")
    raw_data, ticker = load_options_data(args.data_file)

    # Determine strategies to train
    strategies = args.strategies if args.strategies else list(SPECIALIST_REGISTRY.keys())
    logger.info(f"Will train the following strategies: {', '.join(strategies)}")
    
    # Train specialist agents in parallel
    train_specialists(raw_data, ticker, strategies, args)
    
    # Train the manager agent
    train_manager(raw_data, ticker, strategies, args)
    
    # Evaluate the multi-agent system
    evaluate_system(raw_data, ticker, strategies, args)
    
    # Generate trading recommendations
    generate_recommendations(raw_data, ticker, strategies, args)
    
    # Shutdown Ray
    ray.shutdown()
    logger.info("Training complete!")

def train_specialists(raw_data, ticker, strategies, args):
    """Train specialist agents in parallel using Ray"""
    logger.info("Beginning specialist agent training...")
    
    specialist_configs = {}
    
    for strategy in strategies:
        # Create environment config for this strategy
        env_config = {
            "data": raw_data,
            "strategy_name": strategy,
            "is_specialist": True,
        }
        
        # Create training config
        specialist_configs[strategy] = (
            PPOConfig()
            .environment(OptionsBaseEnv, env_config=env_config)
            .training(
                gamma=0.99,
                lr=3e-4,
                kl_coeff=0.2,
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
                clip_param=0.2,
                vf_loss_coeff=0.5,
                entropy_coeff=0.01,
            )
            .resources(num_gpus=args.num_gpus / len(strategies) if args.num_gpus > 0 else 0)
            .rollouts(num_rollout_workers=args.cpu_per_worker)
            .callbacks(SpecialistTrainingCallbacks)
        )
    
    # Run parallel training
    results = {}
    for strategy, config in specialist_configs.items():
        logger.info(f"Training {strategy} specialist...")
        trainer = config.build()
        
        for i in range(args.num_epochs):
            result = trainer.train()
            if i % 10 == 0:
                logger.info(f"{strategy} specialist epoch {i}: reward={result['episode_reward_mean']:.2f}")
                
            # Checkpoint model
            if i % 50 == 0 or i == args.num_epochs - 1:
                checkpoint_path = trainer.save(os.path.join(args.output_dir, f"{ticker}_{strategy}"))
                logger.info(f"Saved {strategy} checkpoint to {checkpoint_path}")
        
        results[strategy] = {
            "final_reward": result['episode_reward_mean'],
            "checkpoint": trainer.save(os.path.join(args.output_dir, f"{ticker}_{strategy}_final"))
        }
        
    # Save specialist results
    with open(os.path.join(args.output_dir, f"{ticker}_specialist_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Specialist training complete!")

def train_manager(raw_data, ticker, strategies, args):
    """Train the manager agent that selects from specialist proposals"""
    logger.info("Beginning manager agent training...")
    
    # Create environment config for manager
    env_config = {
        "data": raw_data,
        "strategies": strategies,
        "is_manager": True,
        "specialist_paths": {
            s: os.path.join(args.output_dir, f"{ticker}_{s}_final") for s in strategies
        }
    }
    
    # Create training config
    manager_config = (
        PPOConfig()
        .environment(OptionsBaseEnv, env_config=env_config)
        .training(
            gamma=0.99,
            lr=1e-4,  # Slightly lower learning rate for manager
            kl_coeff=0.1,
            train_batch_size=2000,
            sgd_minibatch_size=64,
            num_sgd_iter=8,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.02,  # Higher entropy to encourage exploration
        )
        .resources(num_gpus=args.num_gpus if args.num_gpus > 0 else 0)
        .rollouts(num_rollout_workers=max(1, min(4, args.cpu_per_worker)))
        .callbacks(ManagerTrainingCallbacks)
    )
    
    # Train the manager
    trainer = manager_config.build()
    
    for i in range(max(50, args.num_epochs // 2)):  # Manager needs fewer epochs
        result = trainer.train()
        if i % 5 == 0:
            logger.info(f"Manager epoch {i}: reward={result['episode_reward_mean']:.2f}")
        
        # Checkpoint model    
        if i % 20 == 0 or i == max(50, args.num_epochs // 2) - 1:
            checkpoint_path = trainer.save(os.path.join(args.output_dir, f"{ticker}_manager"))
            logger.info(f"Saved manager checkpoint to {checkpoint_path}")
    
    # Save final manager model
    final_checkpoint = trainer.save(os.path.join(args.output_dir, f"{ticker}_manager_final"))
    
    manager_results = {
        "final_reward": result['episode_reward_mean'],
        "checkpoint": final_checkpoint
    }
    
    # Save manager results
    with open(os.path.join(args.output_dir, f"{ticker}_manager_results.json"), "w") as f:
        json.dump(manager_results, f, indent=2)
    
    logger.info("Manager training complete!")

def evaluate_system(raw_data, ticker, strategies, args):
    """Evaluate the complete multi-agent system"""
    logger.info("Evaluating multi-agent system...")
    
    # Load manager agent
    manager_path = os.path.join(args.output_dir, f"{ticker}_manager_final")
    manager_config = PPOConfig().environment(OptionsBaseEnv).framework("torch")
    manager_agent = manager_config.build()
    manager_agent.restore(manager_path)
    
    # Create evaluation environment
    env_config = {
        "data": raw_data,
        "strategies": strategies,
        "is_evaluation": True,
        "specialist_paths": {
            s: os.path.join(args.output_dir, f"{ticker}_{s}_final") for s in strategies
        }
    }
    
    env = OptionsBaseEnv(env_config)
    
    # Run evaluation episodes
    num_episodes = 10
    total_reward = 0
    portfolio_values = []
    trades = []
    
    for episode in range(num_episodes):
        done = False
        obs = env.reset()
        episode_reward = 0
        
        while not done:
            action = manager_agent.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            portfolio_values.append(info.get("portfolio_value", 0))
            
            if info.get("trade_executed", False):
                trades.append(info.get("trade_info", {}))
        
        total_reward += episode_reward
        logger.info(f"Evaluation episode {episode}: reward={episode_reward:.2f}")
    
    avg_reward = total_reward / num_episodes
    sharpe = calculate_sharpe_ratio(portfolio_values)
    win_rate = sum(1 for t in trades if t.get("profit", 0) > 0) / max(1, len(trades))
    
    # Save evaluation results
    results = {
        "ticker": ticker,
        "avg_reward": float(avg_reward),
        "sharpe_ratio": float(sharpe),
        "win_rate": float(win_rate),
        "num_trades": len(trades),
        "trade_details": trades[:20]  # Save sample of trades
    }
    
    with open(os.path.join(args.output_dir, f"{ticker}_evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation complete - Avg Reward: {avg_reward:.2f}, Sharpe: {sharpe:.2f}, Win Rate: {win_rate:.2f}")

def generate_recommendations(raw_data, ticker, strategies, args):
    """Generate trading recommendations from the multi-agent system"""
    logger.info("Generating trading recommendations...")
    
    # Load manager agent
    manager_path = os.path.join(args.output_dir, f"{ticker}_manager_final")
    manager_config = PPOConfig().environment(OptionsBaseEnv).framework("torch")
    manager_agent = manager_config.build()
    manager_agent.restore(manager_path)
    
    # Create recommendation environment
    env_config = {
        "data": raw_data,
        "strategies": strategies,
        "is_recommendation": True,
        "specialist_paths": {
            s: os.path.join(args.output_dir, f"{ticker}_{s}_final") for s in strategies
        },
        "use_latest_data": True
    }
    
    env = OptionsBaseEnv(env_config)
    obs = env.reset()
    
    # Generate recommendations
    num_recommendations = 5
    recommendations = []
    
    for _ in range(num_recommendations):
        action = manager_agent.compute_single_action(obs)
        _, _, _, info = env.step(action)
        
        if info.get("recommendation"):
            recommendations.append(info["recommendation"])
    
    # Format and save recommendations
    formatted_recs = []
    for i, rec in enumerate(recommendations):
        formatted_rec = {
            "rank": i + 1,
            "ticker": ticker,
            "strategy": rec.get("strategy", ""),
            "expected_return": float(rec.get("expected_return", 0)),
            "probability_of_profit": float(rec.get("pop", 0)),
            "risk_reward_ratio": float(rec.get("risk_reward", 0)),
            "expiration": rec.get("expiration", ""),
            "legs": rec.get("legs", [])
        }
        formatted_recs.append(formatted_rec)
    
    # Save recommendations
    with open(os.path.join(args.output_dir, f"{ticker}_recommendations.json"), "w") as f:
        json.dump(formatted_recs, f, indent=2)
    
    logger.info(f"Generated {len(formatted_recs)} recommendations for {ticker}")
    
    # Display top recommendation
    if formatted_recs:
        top = formatted_recs[0]
        logger.info("Top recommendation:")
        logger.info(f"  Strategy: {top['strategy']}")
        logger.info(f"  Expected Return: {top['expected_return']:.2f}")
        logger.info(f"  Probability of Profit: {top['probability_of_profit']:.2f}")
        logger.info(f"  Risk/Reward: {top['risk_reward_ratio']:.2f}")
def extract_options_data(raw_data, trading_day):
    """
    Extract and organize options data for a specific trading day
    
    Args:
        raw_data: Dictionary with market data
        trading_day: Trading day to extract data for
        
    Returns:
        Dictionary with processed options data
    """
    # If raw_data doesn't have this trading day, return empty data
    if trading_day not in raw_data:
        logger.warning(f"No data available for trading day {trading_day}")
        return {
            "underlying_price": 100.0,  # Default value
            "expirations": [],
            "call_options": {},
            "put_options": {},
        }
    
    day_data = raw_data[trading_day]
    
    # Extract underlying price - making sure it's handled robustly
    underlying_price = day_data.get("underlyingPrice", [100.0])[0] if isinstance(day_data.get("underlyingPrice", [100.0]), list) else day_data.get("underlyingPrice", 100.0)
    
    # Process options data
    call_options = {}
    put_options = {}
    expirations = set()
    
    # Check if options data exists in the expected format
    if "expiration" not in day_data or "strike" not in day_data:
        logger.warning(f"Options data not in expected format for {trading_day}")
        return {
            "underlying_price": underlying_price,
            "expirations": [],
            "call_options": {},
            "put_options": {},
        }
    
    # Get arrays of data
    expirations_array = day_data.get("expiration", [])
    strikes_array = day_data.get("strike", [])
    sides_array = day_data.get("side", [])
    
    # Get bid/ask prices
    bid_array = np.array(day_data.get("bid", []))
    ask_array = np.array(day_data.get("ask", []))
    
    # Calculate mid prices (or use other price fields if available)
    if "last" in day_data:
        prices_array = np.array(day_data.get("last", []))
    else:
        prices_array = (bid_array + ask_array) / 2
    
    # Initialize data structures
    unique_expirations = sorted(set(expirations_array))
    
    # Extract options data by expiration and type
    for exp in unique_expirations:
        expirations.add(exp)
        
        # Find all call options for this expiration
        call_indices = [i for i, (exp_i, side) in enumerate(zip(expirations_array, sides_array)) 
                        if exp_i == exp and side.lower() == 'call']
        
        # Find all put options for this expiration
        put_indices = [i for i, (exp_i, side) in enumerate(zip(expirations_array, sides_array)) 
                      if exp_i == exp and side.lower() == 'put']
        
        # Process calls for this expiration
        if call_indices:
            call_strikes = [strikes_array[i] for i in call_indices]
            call_prices = [prices_array[i] for i in call_indices]
            
            # Additional data if available
            call_ivs = [day_data.get("impliedVolatility", [])[i] if "impliedVolatility" in day_data and i < len(day_data["impliedVolatility"]) else 0.3 for i in call_indices]
            call_deltas = [day_data.get("delta", [])[i] if "delta" in day_data and i < len(day_data["delta"]) else 0.5 for i in call_indices]
            
            # Sort by strike
            sorted_indices = np.argsort([float(s) for s in call_strikes])
            call_options[exp] = {
                "strikes": [call_strikes[i] for i in sorted_indices],
                "prices": [call_prices[i] for i in sorted_indices],
                "ivs": [call_ivs[i] for i in sorted_indices],
                "deltas": [call_deltas[i] for i in sorted_indices]
            }
        
        # Process puts for this expiration
        if put_indices:
            put_strikes = [strikes_array[i] for i in put_indices]
            put_prices = [prices_array[i] for i in put_indices]
            
            # Additional data if available
            put_ivs = [day_data.get("impliedVolatility", [])[i] if "impliedVolatility" in day_data and i < len(day_data["impliedVolatility"]) else 0.3 for i in put_indices]
            put_deltas = [day_data.get("delta", [])[i] if "delta" in day_data and i < len(day_data["delta"]) else -0.5 for i in put_indices]
            
            # Sort by strike
            sorted_indices = np.argsort([float(s) for s in put_strikes])
            put_options[exp] = {
                "strikes": [put_strikes[i] for i in sorted_indices],
                "prices": [put_prices[i] for i in sorted_indices],
                "ivs": [put_ivs[i] for i in sorted_indices],
                "deltas": [put_deltas[i] for i in sorted_indices]
            }
    
    # Return processed data
    return {
        "underlying_price": underlying_price,
        "expirations": list(expirations),
        "call_options": call_options,
        "put_options": put_options
    }








class SpecialistTrainingCallbacks(DefaultCallbacks):
    """Callbacks for specialist agent training"""
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Record strategy-specific metrics at episode end"""
        info = episode.last_info_for()
        if info:
            episode.custom_metrics["win_rate"] = info.get("win_rate", 0)
            episode.custom_metrics["trades_executed"] = info.get("trades", 0)
            episode.custom_metrics["avg_pop"] = info.get("avg_pop", 0)

class ManagerTrainingCallbacks(DefaultCallbacks):
    """Callbacks for manager agent training"""
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Record manager-specific metrics at episode end"""
        info = episode.last_info_for()
        if info:
            episode.custom_metrics["portfolio_value"] = info.get("portfolio_value", 0)
            episode.custom_metrics["strategy_diversity"] = info.get("strategy_diversity", 0)
            episode.custom_metrics["opportunity_cost"] = info.get("opportunity_cost", 0)

if __name__ == "__main__":
    main()

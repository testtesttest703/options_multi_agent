import os
import json
import argparse
import logging
import torch
import numpy as np
import ray
import os
import sys
import time
import gc
import psutil
import shutil
# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def check_disk_space(min_gb=1.0):
    """Check if there's enough disk space available"""
    ray_temp_dir = "/home/ubuntu/ray_temp"
    
    # Get disk stats
    stats = shutil.disk_usage(ray_temp_dir)
    free_gb = stats.free / (1024**3)
    
    logger.info(f"Disk space check: {free_gb:.2f} GB free of {stats.total/(1024**3):.2f} GB total")
    
    return free_gb >= min_gb

def clean_ray_temp():
    """Clean up old Ray session directories"""
    ray_temp_dir = "/home/ubuntu/ray_temp"
    
    # Skip if directory doesn't exist
    if not os.path.exists(ray_temp_dir):
        os.makedirs(ray_temp_dir, exist_ok=True)
        return
        
    try:
        # Clean up old Ray sessions
        for item in os.listdir(ray_temp_dir):
            if item.startswith("session_") and item != "session_latest":
                path = os.path.join(ray_temp_dir, item)
                if os.path.isdir(path):
                    logger.info(f"Removing old Ray session: {path}")
                    shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Error cleaning Ray temp directories: {e}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Options Trading with Ray/RLlib")
    parser.add_argument("--data_file", type=str, required=True, help="Path to options data JSON file")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory for saving models and results")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--strategies", type=str, nargs='+', default=None, help="Specific strategies to train")
    parser.add_argument("--cpu_per_worker", type=int, default=1, help="CPUs per worker")
    parser.add_argument("--num_gpus", type=float, default=0, help="GPUs to use for training (can be fractional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logging based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Clean up old Ray sessions to free disk space
    logger.info("Cleaning up old Ray sessions...")
    clean_ray_temp()
    
    # Check disk space
    if not check_disk_space(min_gb=1.0):
        logger.error("Not enough disk space available (less than 1GB). Exiting.")
        return

    # Load the data
    logger.info(f"Loading options data from {args.data_file}")
    start_time = time.time()
    try:
        raw_data, ticker = load_options_data(args.data_file)
        load_time = time.time() - start_time
        logger.info(f"Data loading completed in {load_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Add extensive logging about the loaded data
    logger.info(f"Ticker identified: {ticker}")
    logger.info(f"Memory usage after data loading:")
    log_memory_usage()

    # Log data structure information
    if isinstance(raw_data, dict):
        logger.info(f"Data structure: Dictionary with {len(raw_data)} keys")
        sample_keys = list(raw_data.keys())[:5]
        logger.info(f"Sample dates: {sample_keys}")

        # Log a sample of the nested structure
        if sample_keys and isinstance(raw_data[sample_keys[0]], dict):
            nested_keys = list(raw_data[sample_keys[0]].keys())[:3]
            logger.info(f"Nested structure for {sample_keys[0]}: {nested_keys}")
    else:
        logger.info(f"Data structure: {type(raw_data)}, not a dictionary")

    # Garbage collection to free memory
    logger.info("Running garbage collection...")
    gc.collect()
    log_memory_usage()

    # Initialize Ray with proper settings for 15GB RAM system
    logger.info("Initializing Ray...")
    os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
    
    try:
        ray.init(
            num_cpus=os.cpu_count(),  # Use all available CPUs
            object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB for object store
            _memory=6 * 1024 * 1024 * 1024,  # 6GB total
            num_gpus=args.num_gpus,
            ignore_reinit_error=True,
            include_dashboard=True,  # Enable dashboard now that we have memory
            _temp_dir="/home/ubuntu/ray_temp"  # Use home directory instead of /tmp
        )
        logger.info(f"Ray initialization complete. Available resources: {ray.available_resources()}")
    except Exception as e:
        logger.error(f"Ray initialization failed: {e}")
        logger.info("Trying with backup configuration...")
        try:
            ray.init(
                num_cpus=2,
                object_store_memory=1 * 1024 * 1024 * 1024,  # 1GB
                _memory=3 * 1024 * 1024 * 1024,  # 3GB total
                ignore_reinit_error=True,
                _temp_dir="/home/ubuntu/ray_temp"
            )
            logger.info("Ray initialized with backup configuration")
        except Exception as e2:
            logger.error(f"Backup Ray initialization also failed: {e2}")
            logger.error("Cannot proceed without Ray. Exiting.")
            return

    # Determine strategies to train
    logger.info("Determining strategies to train...")
    strategies = args.strategies if args.strategies else list(SPECIALIST_REGISTRY.keys())
    logger.info(f"Will train the following strategies: {', '.join(strategies)}")

    # Log memory and timing before specialist training
    logger.info("Preparing to train specialists...")
    log_memory_usage()

    # Train specialist agents in parallel
    logger.info("Starting specialist training...")
    start_time = time.time()
    train_specialists(raw_data, ticker, strategies, args)
    specialist_time = time.time() - start_time
    logger.info(f"Specialist training completed in {specialist_time:.2f} seconds")
    log_memory_usage()

    # Train the manager agent
    logger.info("Starting manager training...")
    start_time = time.time()
    train_manager(raw_data, ticker, strategies, args)
    manager_time = time.time() - start_time
    logger.info(f"Manager training completed in {manager_time:.2f} seconds")
    log_memory_usage()

    # Evaluate the multi-agent system
    logger.info("Starting system evaluation...")
    evaluate_system(raw_data, ticker, strategies, args)

    # Generate trading recommendations
    logger.info("Generating recommendations...")
    generate_recommendations(raw_data, ticker, strategies, args)

    # Shutdown Ray
    logger.info("Shutting down Ray...")
    ray.shutdown()
    logger.info("Training complete!")

def train_specialists(raw_data, ticker, strategies, args):
    """Train specialist agents in parallel using Ray"""
    logger.info("Beginning specialist agent training...")

    specialist_configs = {}

    # Log creation of each specialist config
    for strategy in strategies:
        # Create environment config for this strategy
        logger.info(f"Creating environment config for {strategy} specialist...")
        env_config = {
            "data": raw_data,
            "strategy_name": strategy,
            "is_specialist": True,
        }

        # Create training config with UPDATED API for Ray
        logger.info(f"Creating PPO config for {strategy} specialist...")
        config = (
            PPOConfig()
            .environment(OptionsBaseEnv, env_config=env_config)
            .training(
                gamma=0.99,
                lr=3e-4,
                kl_coeff=0.2,
                train_batch_size=4000,
                num_sgd_iter=10,
                clip_param=0.2,
                vf_loss_coeff=0.5,
                entropy_coeff=0.01,
            )
            .resources(num_gpus=args.num_gpus / len(strategies) if args.num_gpus > 0 else 0)
            # Fixed: Using env_runners instead of rollouts
            .env_runners(num_env_runners=args.cpu_per_worker)
            .callbacks(SpecialistTrainingCallbacks)
        )
        
        # Set minibatch size separately
        config.minibatch_size = 128
        
        specialist_configs[strategy] = config
        logger.info(f"Config for {strategy} specialist created successfully")

    # Run parallel training
    results = {}
    for strategy, config in specialist_configs.items():
        logger.info(f"Training {strategy} specialist...")
        logger.info(f"Building trainer for {strategy}...")
        trainer = config.build()
        logger.info(f"Trainer for {strategy} built successfully")

        for i in range(args.num_epochs):
            logger.info(f"Starting {strategy} epoch {i}...")
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
    logger.info("Saving specialist results...")
    with open(os.path.join(args.output_dir, f"{ticker}_specialist_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Specialist training complete!")

def train_manager(raw_data, ticker, strategies, args):
    """Train the manager agent that selects from specialist proposals"""
    logger.info("Beginning manager agent training...")

    # Create environment config for manager
    logger.info("Creating manager environment config...")
    env_config = {
        "data": raw_data,
        "strategies": strategies,
        "is_manager": True,
        "specialist_paths": {
            s: os.path.join(args.output_dir, f"{ticker}_{s}_final") for s in strategies
        }
    }
    logger.info(f"Manager will use specialists from paths: {list(env_config['specialist_paths'].values())}")

    # Create training config with UPDATED API for Ray
    logger.info("Creating manager PPO config...")
    manager_config = (
        PPOConfig()
        .environment(OptionsBaseEnv, env_config=env_config)
        .training(
            gamma=0.99,
            lr=1e-4,  # Slightly lower learning rate for manager
            kl_coeff=0.1,
            train_batch_size=2000,
            num_sgd_iter=8,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.02,  # Higher entropy to encourage exploration
        )
        .resources(num_gpus=args.num_gpus if args.num_gpus > 0 else 0)
        # Fixed: Using env_runners instead of rollouts
        .env_runners(num_env_runners=max(1, min(4, args.cpu_per_worker)))
        .callbacks(ManagerTrainingCallbacks)
    )
    
    # Set minibatch size separately
    manager_config.minibatch_size = 64

    # Train the manager
    logger.info("Building manager trainer...")
    trainer = manager_config.build()
    logger.info("Manager trainer built successfully")

    for i in range(max(50, args.num_epochs // 2)):  # Manager needs fewer epochs
        logger.info(f"Starting manager epoch {i}...")
        result = trainer.train()
        if i % 5 == 0:
            logger.info(f"Manager epoch {i}: reward={result['episode_reward_mean']:.2f}")

        # Checkpoint model
        if i % 20 == 0 or i == max(50, args.num_epochs // 2) - 1:
            checkpoint_path = trainer.save(os.path.join(args.output_dir, f"{ticker}_manager"))
            logger.info(f"Saved manager checkpoint to {checkpoint_path}")

    # Save final manager model
    logger.info("Saving final manager model...")
    final_checkpoint = trainer.save(os.path.join(args.output_dir, f"{ticker}_manager_final"))

    manager_results = {
        "final_reward": result['episode_reward_mean'],
        "checkpoint": final_checkpoint
    }

    # Save manager results
    logger.info("Saving manager results...")
    with open(os.path.join(args.output_dir, f"{ticker}_manager_results.json"), "w") as f:
        json.dump(manager_results, f, indent=2)

    logger.info("Manager training complete!")

def evaluate_system(raw_data, ticker, strategies, args):
    """Evaluate the complete multi-agent system"""
    logger.info("Evaluating multi-agent system...")

    # Load manager agent
    logger.info(f"Loading manager agent from checkpoint...")
    manager_path = os.path.join(args.output_dir, f"{ticker}_manager_final")
    manager_config = PPOConfig().environment(OptionsBaseEnv).framework("torch")
    manager_agent = manager_config.build()
    manager_agent.restore(manager_path)
    logger.info(f"Manager agent loaded successfully")

    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    env_config = {
        "data": raw_data,
        "strategies": strategies,
        "is_evaluation": True,
        "specialist_paths": {
            s: os.path.join(args.output_dir, f"{ticker}_{s}_final") for s in strategies
        }
    }

    env = OptionsBaseEnv(env_config)
    logger.info("Evaluation environment created successfully")

    # Run evaluation episodes
    num_episodes = 10
    total_reward = 0
    portfolio_values = []
    trades = []

    for episode in range(num_episodes):
        logger.info(f"Starting evaluation episode {episode}...")
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
    logger.info("Saving evaluation results...")
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
    logger.info("Loading manager agent for recommendations...")
    manager_path = os.path.join(args.output_dir, f"{ticker}_manager_final")
    manager_config = PPOConfig().environment(OptionsBaseEnv).framework("torch")
    manager_agent = manager_config.build()
    manager_agent.restore(manager_path)

    # Create recommendation environment
    logger.info("Creating recommendation environment...")
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
    logger.info("Recommendation environment created successfully")

    obs = env.reset()
    logger.info("Environment reset complete, generating recommendations...")

    # Generate recommendations
    num_recommendations = 5
    recommendations = []

    for i in range(num_recommendations):
        logger.info(f"Generating recommendation {i+1}/{num_recommendations}...")
        action = manager_agent.compute_single_action(obs)
        _, _, _, info = env.step(action)

        if info.get("recommendation"):
            recommendations.append(info["recommendation"])
            logger.info(f"Recommendation {i+1} generated successfully")
        else:
            logger.warning(f"No recommendation generated in step {i+1}")

    # Format and save recommendations
    logger.info("Formatting recommendations...")
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
    logger.info(f"Saving {len(formatted_recs)} recommendations...")
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

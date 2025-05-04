#!/usr/bin/env python3
"""
Lightweight version of the main script that uses minimal memory
"""
import os
import sys
import json
import logging
import time
import traceback
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("LightweightRun")

def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight options trading script")
    parser.add_argument("--data_file", type=str, required=True, help="Path to options data JSON file")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory for saving models and results")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def log_step(step_name):
    """Helper to log each step with a clear separator"""
    logger.info("=" * 40)
    logger.info(f"STEP: {step_name}")
    logger.info("=" * 40)
    return time.time()

def main():
    args = parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Step 1: Load data
        start_time = log_step("Loading data")
        from utils.data_loader import load_options_data
        options_data, ticker = load_options_data(args.data_file)
        logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
        
        # Step 2: Initialize Ray with minimal memory settings
        start_time = log_step("Initializing Ray")
        import ray
        os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
        ray.init(
            num_cpus=1,
            object_store_memory=100 * 1024 * 1024,  # 100 MB
            _memory=300 * 1024 * 1024,              # 300 MB
            ignore_reinit_error=True
        )
        logger.info(f"Ray initialized in {time.time() - start_time:.2f} seconds")
        
        # Step 3: Create minimal environment for testing
        start_time = log_step("Creating environment")
        from env.generic_env import OptionsBaseEnv
        
        env_config = {
            "data": options_data,
            "strategy_name": "IRON CONDOR",  # Simplest strategy 
            "is_specialist": True,
        }
        
        env = OptionsBaseEnv(env_config)
        logger.info(f"Environment created in {time.time() - start_time:.2f} seconds")
        
        # Step 4: Reset and test the environment
        start_time = log_step("Testing environment")
        obs = env.reset()
        logger.info("Environment reset successful")
        
        # Try a sample step
        action = env.action_space.sample()
        logger.info(f"Taking random action: {action}")
        next_obs, reward, done, info = env.step(action)
        logger.info(f"Step completed: reward={reward}, done={done}")
        
        # Step 5: Create minimal PPO config for testing  
        start_time = log_step("Creating minimal PPO config")
        from ray.rllib.algorithms.ppo import PPOConfig
        
        trainer_config = (
            PPOConfig()
            .environment(OptionsBaseEnv, env_config=env_config)
            .framework("torch")
            .rollouts(num_rollout_workers=0)  # No workers to save memory
            .training(
                train_batch_size=200,  # Small batch size
                sgd_minibatch_size=100,
                num_sgd_iter=5,
            )
            .resources(num_gpus=0)
        )
        
        # Step 6: Build a minimal trainer
        start_time = log_step("Building trainer")
        trainer = trainer_config.build()
        logger.info(f"Trainer built in {time.time() - start_time:.2f} seconds")
        
        # Step 7: Do a training step
        start_time = log_step("Doing one training step")
        result = trainer.train()
        logger.info(f"Training step completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Result: {result}")
        
        # Step 8: Save a checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"{ticker}_test")
        checkpoint_path = trainer.save(checkpoint_dir)
        logger.info(f"Saved checkpoint to: {checkpoint_path}")
        
        # Shutdown Ray
        ray.shutdown()
        logger.info("Run completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.error(traceback.format_exc())
        
        # Try to shutdown Ray if it was initialized
        try:
            ray.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()

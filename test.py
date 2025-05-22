import subprocess
import os

# Get the absolute path for the models directory
models_dir = os.path.abspath("models")

# Make sure models directory exists
os.makedirs(models_dir, exist_ok=True)

# Run script with the flags to solve stalling
subprocess.run([
    "python3", "main.py",
    "--data_file", "/home/ubuntu/AAPL_all_trading_days_2020-01-01_to_2025-04-09_cached_data_by_date.json",
    "--output_dir", models_dir,
    "--num_epochs", "1",
    "--cpu_per_worker", "1"  # Limit CPU resources to prevent overload
])

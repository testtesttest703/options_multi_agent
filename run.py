#!/usr/bin/env python3
"""
Wrapper script to run the options multi-agent system with correct import paths.
"""
import os
import sys
import subprocess

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Pass all command line arguments to main.py
    args = sys.argv[1:]
    subprocess.run([sys.executable, "main.py"] + args)

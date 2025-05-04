#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import signal

# Set timeout in seconds (adjust as needed)
TIMEOUT = 300  # 5 minutes

def run_with_timeout():
    proc = subprocess.Popen(
        [sys.executable, "main.py"] + sys.argv[1:],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # Define timeout handler
    def timeout_handler():
        print("\n\nTIMEOUT: Process took too long, terminating...")
        proc.terminate()
        # Wait a bit and kill if still running
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Process still running, sending SIGKILL...")
            proc.kill()
    
    # Start timeout timer
    timer = threading.Timer(TIMEOUT, timeout_handler)
    timer.start()
    
    try:
        # Wait for process to complete
        proc.wait()
        # If we get here, cancel the timeout
        timer.cancel()
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected, terminating process...")
        timer.cancel()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == "__main__":
    run_with_timeout()

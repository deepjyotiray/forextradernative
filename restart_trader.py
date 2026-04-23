#!/usr/bin/env python3
"""
Quick restart script for the auto trader
"""
import requests
import time
import subprocess
import sys

def stop_trader():
    """Stop the running trader"""
    try:
        response = requests.post("http://127.0.0.1:8899/shutdown", timeout=5)
        print("Shutdown request sent")
        time.sleep(3)
    except:
        print("No running trader found or already stopped")

def start_trader():
    """Start the trader"""
    try:
        # Start the trader process
        subprocess.Popen([sys.executable, "auto_trader.py"], 
                        cwd="c:\\Users\\deepj\\PycharmProjects\\FastAPIProject")
        print("Trader started successfully")
        print("Dashboard: http://127.0.0.1:8899")
        return True
    except Exception as e:
        print(f"Failed to start trader: {e}")
        return False

if __name__ == "__main__":
    print("Restarting Auto Trader...")
    stop_trader()
    time.sleep(2)
    if start_trader():
        print("Restart complete!")
    else:
        print("Restart failed!")
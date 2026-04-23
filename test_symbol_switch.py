#!/usr/bin/env python3
"""
Test script to verify symbol switching functionality.
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8899"

def test_symbol_switch():
    print("Testing symbol switching functionality...")
    
    # Get initial status
    print("\n1. Getting initial status...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            status = response.json()
            current_symbol = status.get("symbol", "unknown")
            print(f"   Current symbol: {current_symbol}")
            available_symbols = status.get("available_symbols", [])
            print(f"   Available symbols: {available_symbols}")
        else:
            print(f"   Failed to get status: {response.status_code}")
            return
    except Exception as e:
        print(f"   Error getting status: {e}")
        return
    
    # Test switching to a different symbol
    test_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    for test_symbol in test_symbols:
        if test_symbol in available_symbols and test_symbol != current_symbol:
            print(f"\n2. Switching to {test_symbol}...")
            try:
                response = requests.post(f"{BASE_URL}/symbol/{test_symbol}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"   Switch response: {result}")
                    
                    # Wait a moment for the change to take effect
                    time.sleep(2)
                    
                    # Check status again
                    print(f"   Checking status after switch...")
                    status_response = requests.get(f"{BASE_URL}/status")
                    if status_response.status_code == 200:
                        new_status = status_response.json()
                        new_symbol = new_status.get("symbol", "unknown")
                        print(f"   New symbol in status: {new_symbol}")
                        
                        if new_symbol == test_symbol:
                            print(f"   ✅ SUCCESS: Symbol correctly switched to {test_symbol}")
                        else:
                            print(f"   ❌ FAILED: Expected {test_symbol}, got {new_symbol}")
                    else:
                        print(f"   Failed to get status after switch: {status_response.status_code}")
                else:
                    print(f"   Failed to switch symbol: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"   Error switching symbol: {e}")
            
            # Switch back to original symbol
            print(f"\n3. Switching back to {current_symbol}...")
            try:
                response = requests.post(f"{BASE_URL}/symbol/{current_symbol}")
                if response.status_code == 200:
                    time.sleep(2)
                    status_response = requests.get(f"{BASE_URL}/status")
                    if status_response.status_code == 200:
                        final_status = status_response.json()
                        final_symbol = final_status.get("symbol", "unknown")
                        print(f"   Final symbol: {final_symbol}")
                        if final_symbol == current_symbol:
                            print(f"   ✅ SUCCESS: Switched back to {current_symbol}")
                        else:
                            print(f"   ❌ FAILED: Expected {current_symbol}, got {final_symbol}")
            except Exception as e:
                print(f"   Error switching back: {e}")
            
            break
    else:
        print(f"\n   No suitable test symbol found. Current: {current_symbol}, Available: {available_symbols}")

if __name__ == "__main__":
    test_symbol_switch()
#!/usr/bin/env python3
"""
Diagnostic script to check current trading system status.
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8899"

def diagnose_status():
    print("🔍 Diagnosing trading system status...")
    
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code != 200:
            print(f"❌ Failed to get status: HTTP {response.status_code}")
            return
        
        status = response.json()
        
        print(f"\n📊 Current Status:")
        print(f"   Symbol: {status.get('symbol', 'unknown')}")
        print(f"   Available Symbols: {status.get('available_symbols', [])}")
        print(f"   MT5 Connected: {status.get('mt5_connected', False)}")
        print(f"   Trading Enabled: {status.get('enabled', False)}")
        print(f"   Strategy: {status.get('strategy', 'unknown')}")
        
        # Check if there are any recent logs about symbol changes
        logs = status.get('log', [])
        symbol_logs = [log for log in logs if 'symbol' in log.get('msg', '').lower()]
        
        if symbol_logs:
            print(f"\n📝 Recent Symbol-related Logs:")
            for log in symbol_logs[-5:]:  # Show last 5 symbol-related logs
                print(f"   [{log.get('time', 'unknown')}] {log.get('tag', 'LOG')}: {log.get('msg', '')}")
        else:
            print(f"\n📝 No recent symbol-related logs found")
        
        # Check tick data
        tick = status.get('tick', {})
        if tick:
            print(f"\n📈 Current Tick Data:")
            print(f"   Bid: {tick.get('bid', 'N/A')}")
            print(f"   Ask: {tick.get('ask', 'N/A')}")
            print(f"   Spread: {tick.get('spread', 'N/A')}")
            print(f"   Time: {tick.get('time', 'N/A')}")
        else:
            print(f"\n📈 No tick data available")
        
        # Check account info
        account = status.get('account', {})
        if account:
            print(f"\n💰 Account Info:")
            print(f"   Balance: ${account.get('balance', 'N/A')}")
            print(f"   Equity: ${account.get('equity', 'N/A')}")
            print(f"   Server: {account.get('server', 'N/A')}")
        
        print(f"\n✅ Diagnosis complete. If symbol is stuck on XAUUSD, try:")
        print(f"   1. POST to {BASE_URL}/symbol/EURUSD")
        print(f"   2. Check logs for any errors")
        print(f"   3. Restart the trading system if needed")
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")

if __name__ == "__main__":
    diagnose_status()
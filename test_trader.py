import time
import json

def test_trader_status():
    """Test if the trader is running and responding"""
    try:
        import urllib.request
        
        # Test the status endpoint
        with urllib.request.urlopen("http://127.0.0.1:8899/status", timeout=5) as response:
            data = json.loads(response.read().decode())
            
        print("[OK] Trader is running successfully!")
        print(f"   - MT5 Connected: {data.get('mt5_connected', False)}")
        print(f"   - Strategy: {data.get('strategy', 'Unknown')}")
        print(f"   - Trading Enabled: {data.get('enabled', False)}")
        print(f"   - Market Open: {data.get('market_open', False)}")
        print(f"   - Current Price: {data.get('tick', {}).get('bid', 'N/A')}")
        print(f"   - Dashboard: http://127.0.0.1:8899")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Trader not responding: {e}")
        return False

if __name__ == "__main__":
    print("Testing trader status...")
    time.sleep(2)  # Give it a moment to start
    test_trader_status()
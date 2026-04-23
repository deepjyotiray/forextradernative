#!/usr/bin/env python3
"""
Test XGBoost Dashboard Integration
Quick test to verify XGBoost predictions are working in the dashboard.
"""
import json
import requests
import time

def test_xgb_dashboard():
    """Test XGBoost integration in the dashboard."""
    print("🧪 Testing XGBoost Dashboard Integration...")
    
    try:
        # Test the status endpoint
        response = requests.get("http://localhost:8000/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Status endpoint failed: {response.status_code}")
            return
        
        data = response.json()
        
        # Check XGBoost model status
        xgb_info = data.get("xgb", {})
        print(f"\n📊 XGBoost Model Status:")
        print(f"   Trained: {xgb_info.get('trained', False)}")
        print(f"   Training Data: {xgb_info.get('trades', 0)} trades")
        
        if xgb_info.get("trained"):
            features = xgb_info.get("features", [])
            if features:
                print(f"   Top Features:")
                for i, feature in enumerate(features[:5], 1):
                    importance = feature.get("importance", 0) * 100
                    print(f"     {i}. {feature.get('name', 'Unknown')}: {importance:.1f}%")
        
        # Check live predictions
        xgb_live = data.get("xgb_live_prediction", {})
        print(f"\n🔮 Live Predictions:")
        if xgb_live.get("available"):
            buy_prob = xgb_live.get("buy_probability", 0) * 100
            sell_prob = xgb_live.get("sell_probability", 0) * 100
            sentiment = xgb_live.get("sentiment", "UNKNOWN")
            
            print(f"   BUY Win Probability: {buy_prob:.1f}%")
            print(f"   SELL Win Probability: {sell_prob:.1f}%")
            print(f"   Market Sentiment: {sentiment}")
            print(f"   Model Confidence: {xgb_live.get('model_confidence', 'UNKNOWN')}")
            
            # Show current market features
            features_used = xgb_live.get("features_used", {})
            print(f"\n📈 Current Market Features:")
            for key, value in features_used.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
        else:
            reason = xgb_live.get("reason", "Unknown error")
            print(f"   ❌ Unavailable: {reason}")
        
        # Check current market conditions
        print(f"\n🌍 Current Market Conditions:")
        tick = data.get("tick", {})
        regime = data.get("regime", {})
        bias = data.get("bias", {})
        indicators = data.get("indicators", {})
        
        print(f"   Price: {tick.get('bid', 0):.2f}")
        print(f"   Spread: {tick.get('spread', 0):.2f}")
        print(f"   Session: {data.get('session', 'Unknown')}")
        print(f"   Regime: {regime.get('state', 'Unknown')}")
        print(f"   Bias: {bias.get('direction', 'Unknown')} ({(bias.get('confidence', 0)*100):.0f}%)")
        print(f"   ATR: {indicators.get('atr', 0):.3f}")
        print(f"   RSI: {indicators.get('rsi', 50):.1f}")
        
        # Check if trading is enabled
        enabled = data.get("enabled", False)
        mt5_connected = data.get("mt5_connected", False)
        
        print(f"\n⚡ Trading Status:")
        print(f"   Trading Enabled: {'✅' if enabled else '❌'}")
        print(f"   MT5 Connected: {'✅' if mt5_connected else '❌'}")
        
        # Check open positions
        open_trades = data.get("trades", {}).get("open_trades", [])
        print(f"   Open Positions: {len(open_trades)}")
        
        if open_trades:
            print(f"   Recent Positions:")
            for trade in open_trades[:3]:  # Show first 3
                ticket = trade.get("ticket", "Unknown")
                direction = trade.get("direction", "Unknown")
                pnl = trade.get("live_pnl", 0)
                confidence = trade.get("confidence", 0)
                print(f"     #{ticket}: {direction} | P&L: ${pnl:+.2f} | XGB: {confidence*100:.0f}%")
        
        print(f"\n✅ XGBoost Dashboard Integration Test Complete!")
        print(f"🌐 Dashboard URL: http://localhost:8000")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to trading service. Make sure auto_trader.py is running.")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_xgb_dashboard()
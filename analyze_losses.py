#!/usr/bin/env python3
"""
Analyze trading losses around 6:30 PM IST today
"""
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Dict
import pandas as pd

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

def load_trade_history():
    """Load trade history from JSON file"""
    try:
        with open('trade_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("trade_history.json not found")
        return []

def analyze_database_trades():
    """Analyze trades from the database"""
    try:
        conn = sqlite3.connect('orders.db')
        conn.row_factory = sqlite3.Row
        
        # Get today's date in IST
        today_ist = datetime.now(IST).strftime('%Y-%m-%d')
        
        # Get all trades from today
        cursor = conn.execute("""
            SELECT * FROM orders 
            WHERE substr(close_time_ist, 1, 10) = ? OR substr(open_time_ist, 1, 10) = ?
            ORDER BY open_time DESC
        """, (today_ist, today_ist))
        
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return trades
    except Exception as e:
        print(f"Error accessing database: {e}")
        return []

def parse_trade_time(time_str):
    """Parse trade time string to datetime"""
    try:
        if 'T' in time_str:
            return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        return datetime.fromisoformat(time_str)
    except:
        return None

def analyze_losses_around_630pm():
    """Analyze losses around 6:30 PM IST"""
    print("=== ANALYZING TRADING LOSSES AROUND 6:30 PM IST ===\n")
    
    # Load trade history
    trade_history = load_trade_history()
    db_trades = analyze_database_trades()
    
    # Get today's date
    today = datetime.now(IST).date()
    target_time = datetime.combine(today, datetime.min.time().replace(hour=18, minute=30))
    target_time = IST.localize(target_time)
    
    print(f"Target time: {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Analyzing trades from: {(target_time - timedelta(hours=2)).strftime('%H:%M')} to {(target_time + timedelta(hours=2)).strftime('%H:%M')} IST\n")
    
    # Analyze JSON trade history
    print("=== TRADE HISTORY ANALYSIS ===")
    relevant_trades = []
    total_pnl = 0
    losing_trades = 0
    winning_trades = 0
    
    for trade in trade_history:
        trade_time = parse_trade_time(trade.get('time', ''))
        if not trade_time:
            continue
            
        # Convert to IST
        trade_time_ist = trade_time.astimezone(IST)
        
        # Check if trade is from today and around target time
        if trade_time_ist.date() == today:
            time_diff = abs((trade_time_ist - target_time).total_seconds() / 3600)  # hours
            
            if time_diff <= 3:  # Within 3 hours of 6:30 PM
                relevant_trades.append({
                    'ticket': trade.get('ticket'),
                    'time_ist': trade_time_ist.strftime('%H:%M:%S'),
                    'pnl': trade.get('pnl', 0),
                    'direction': trade.get('direction'),
                    'entry_price': trade.get('entry_price'),
                    'exit_price': trade.get('exit_price'),
                    'comment': trade.get('comment', ''),
                    'features': trade.get('features', {})
                })
                
                pnl = trade.get('pnl', 0)
                total_pnl += pnl
                if pnl < 0:
                    losing_trades += 1
                else:
                    winning_trades += 1
    
    # Sort by time
    relevant_trades.sort(key=lambda x: x['time_ist'])
    
    print(f"Found {len(relevant_trades)} trades around 6:30 PM IST")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Winning trades: {winning_trades}")
    print(f"Losing trades: {losing_trades}")
    print(f"Win rate: {(winning_trades/(winning_trades+losing_trades)*100):.1f}%" if (winning_trades+losing_trades) > 0 else "N/A")
    print()
    
    # Show detailed trades
    print("=== DETAILED TRADE BREAKDOWN ===")
    cumulative_pnl = 0
    
    for i, trade in enumerate(relevant_trades, 1):
        cumulative_pnl += trade['pnl']
        status = "WIN" if trade['pnl'] > 0 else "LOSS"
        
        print(f"{i:2d}. {trade['time_ist']} | {trade['direction']:4s} | "
              f"Entry: {trade['entry_price']:8.2f} | Exit: {trade['exit_price']:8.2f} | "
              f"PnL: ${trade['pnl']:6.2f} | Cumulative: ${cumulative_pnl:6.2f} | {status}")
        
        if trade['comment']:
            print(f"     Comment: {trade['comment']}")
        
        # Show key features if available
        features = trade.get('features', {})
        if features:
            key_features = []
            for key in ['regime', 'session', 'rsi', 'atr_ratio', 'bias_conf']:
                if key in features:
                    key_features.append(f"{key}: {features[key]}")
            if key_features:
                print(f"     Features: {', '.join(key_features)}")
        print()
    
    # Analyze patterns in losing trades
    print("=== LOSS ANALYSIS ===")
    losses = [t for t in relevant_trades if t['pnl'] < 0]
    
    if losses:
        total_loss = sum(t['pnl'] for t in losses)
        avg_loss = total_loss / len(losses)
        
        print(f"Total losses: ${total_loss:.2f}")
        print(f"Average loss per trade: ${avg_loss:.2f}")
        print(f"Largest single loss: ${min(t['pnl'] for t in losses):.2f}")
        
        # Analyze stop loss hits
        sl_hits = [t for t in losses if 'sl' in t['comment'].lower()]
        print(f"Stop loss hits: {len(sl_hits)} out of {len(losses)} losses")
        
        # Analyze by direction
        sell_losses = [t for t in losses if t['direction'] == 'SELL']
        buy_losses = [t for t in losses if t['direction'] == 'BUY']
        print(f"SELL losses: {len(sell_losses)} (${sum(t['pnl'] for t in sell_losses):.2f})")
        print(f"BUY losses: {len(buy_losses)} (${sum(t['pnl'] for t in buy_losses):.2f})")
        
        # Time analysis
        print("\n=== TIME PATTERN ANALYSIS ===")
        time_buckets = {}
        for trade in losses:
            hour = int(trade['time_ist'].split(':')[0])
            time_buckets[hour] = time_buckets.get(hour, 0) + trade['pnl']
        
        for hour in sorted(time_buckets.keys()):
            print(f"{hour:2d}:00-{hour:2d}:59 IST: ${time_buckets[hour]:.2f}")
    
    # Database analysis
    print("\n=== DATABASE ANALYSIS ===")
    if db_trades:
        today_trades = []
        for trade in db_trades:
            if trade.get('close_time_ist'):
                close_time = datetime.fromisoformat(trade['close_time_ist'])
                time_diff = abs((close_time - target_time).total_seconds() / 3600)
                if time_diff <= 3:
                    today_trades.append(trade)
        
        if today_trades:
            db_total_pnl = sum(t.get('final_pnl', 0) for t in today_trades if t.get('final_pnl'))
            db_losses = [t for t in today_trades if t.get('final_pnl', 0) < 0]
            
            print(f"Database trades around 6:30 PM: {len(today_trades)}")
            print(f"Database total PnL: ${db_total_pnl:.2f}")
            print(f"Database losses: {len(db_losses)}")
            
            if db_losses:
                print("\nDatabase loss details:")
                for trade in db_losses:
                    print(f"  Ticket {trade['ticket']}: ${trade.get('final_pnl', 0):.2f} - {trade.get('close_reason', 'N/A')}")
    
    # Market condition analysis
    print("\n=== MARKET CONDITION ANALYSIS ===")
    if relevant_trades:
        regimes = {}
        sessions = {}
        
        for trade in relevant_trades:
            features = trade.get('features', {})
            regime = features.get('regime', 'UNKNOWN')
            session = features.get('session', 'UNKNOWN')
            
            regimes[regime] = regimes.get(regime, {'count': 0, 'pnl': 0})
            regimes[regime]['count'] += 1
            regimes[regime]['pnl'] += trade['pnl']
            
            sessions[session] = sessions.get(session, {'count': 0, 'pnl': 0})
            sessions[session]['count'] += 1
            sessions[session]['pnl'] += trade['pnl']
        
        print("Performance by Market Regime:")
        for regime, data in regimes.items():
            print(f"  {regime}: {data['count']} trades, ${data['pnl']:.2f} PnL")
        
        print("\nPerformance by Trading Session:")
        for session, data in sessions.items():
            print(f"  {session}: {data['count']} trades, ${data['pnl']:.2f} PnL")

if __name__ == "__main__":
    analyze_losses_around_630pm()
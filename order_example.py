"""
Order Integration Example - Shows how to use the new order database system.
This demonstrates the complete workflow from order creation to tracking.
"""
from engine.order_database import OrderDatabase
from engine.session_detector import SessionDetector
from engine.trade_manager import TradeManager
from engine.mt5_bridge import MT5Bridge
import config as cfg


class OrderIntegrationExample:
    """Example showing complete order management workflow."""
    
    def __init__(self):
        self.order_db = OrderDatabase()
        self.session_detector = SessionDetector()
        
        # Initialize MT5 and TradeManager (if available)
        self.mt5_bridge = None
        self.trade_manager = None
        try:
            self.mt5_bridge = MT5Bridge()
            if self.mt5_bridge.connect():
                self.trade_manager = TradeManager(self.mt5_bridge)
        except Exception as e:
            print(f"MT5 not available: {e}")
    
    def create_sample_order(self, strategy_name: str = "EXAMPLE"):
        """Create a sample order with full context."""
        # Get session information
        session, phase = self.session_detector.get_current_session()
        session_info = self.session_detector.get_session_info()
        
        # Sample order parameters
        ticket = 999999  # Example ticket
        direction = "BUY"
        volume = 0.1
        entry_price = 2650.50
        sl = 2648.00
        tp = 2655.00
        sl_distance = abs(entry_price - sl)
        
        # Enhanced features with session context
        features = {
            "session_info": session_info,
            "spread": 0.25,
            "volatility": "MEDIUM",
            "trend": "BULLISH",
            "confluence_score": 8.5,
            "risk_reward": 2.0
        }
        
        # Store order in database
        success = self.order_db.store_order(
            ticket=ticket,
            direction=direction,
            volume=volume,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            sl_distance=sl_distance,
            strategy=strategy_name,
            confidence=85.0,
            reason="Example order with full context",
            scalp=False,
            be_trigger=1.0,
            timeout=300,
            features=features,
            session_type=session,
            market_phase=phase
        )
        
        if success:
            print(f"✅ Order {ticket} stored successfully")
            print(f"   Session: {session} ({phase})")
            print(f"   Strategy: {strategy_name}")
            print(f"   Entry: {entry_price}, SL: {sl}, TP: {tp}")
            
            # If TradeManager is available, register there too
            if self.trade_manager:
                self.trade_manager.register_trade(
                    ticket, direction, volume, entry_price, sl, tp, sl_distance,
                    strategy_name, 85.0, "Example order", False, 1.0, 300, features,
                    session, phase
                )
                print(f"   Also registered in TradeManager")
        else:
            print(f"❌ Failed to store order {ticket}")
        
        return success
    
    def simulate_order_lifecycle(self, ticket: int = 999999):
        """Simulate complete order lifecycle."""
        print(f"\n📊 Simulating lifecycle for order {ticket}")
        
        # 1. Update live PnL (simulating MT5 updates)
        pnl_updates = [5.50, 12.30, -2.10, 8.75, 15.20]
        
        for i, pnl in enumerate(pnl_updates):
            self.order_db.update_live_pnl(ticket, pnl)
            print(f"   Update {i+1}: Live PnL = ${pnl}")
        
        # 2. Update management flags
        self.order_db.update_management_flags(ticket, sl_breakeven=True)
        print(f"   ✅ Moved to breakeven")
        
        self.order_db.update_management_flags(ticket, partial_closed=True)
        print(f"   ✅ Partial close executed")
        
        # 3. Close order with final PnL
        final_pnl = 18.50
        self.order_db.close_order(ticket, final_pnl, "TP_HIT", swap=-0.25, commission=-0.15)
        print(f"   ✅ Order closed: Final PnL = ${final_pnl}")
    
    def show_statistics(self):
        """Display current statistics."""
        print(f"\n📈 Current Statistics:")
        
        # Today's stats
        today_stats = self.order_db.get_today_stats()
        print(f"   Today: {today_stats['trades']} trades, ${today_stats['total_pnl']} PnL")
        print(f"   W/L: {today_stats['wins']}/{today_stats['losses']}")
        print(f"   Open: {today_stats['open_count']} positions, ${today_stats['open_pnl']} floating")
        
        # Strategy performance
        strategy_perf = self.order_db.get_strategy_performance(30)
        if strategy_perf:
            print(f"\n   Strategy Performance (30 days):")
            for strat in strategy_perf[:3]:  # Top 3
                print(f"   {strat['strategy']}: {strat['trades']} trades, "
                      f"${strat['total_pnl']}, {strat['win_rate']}% WR")
    
    def export_data(self):
        """Export current data snapshot."""
        print(f"\n💾 Exporting data snapshot...")
        filepath = self.order_db.export_snapshot()
        print(f"   Exported to: {filepath}")
        return filepath
    
    def show_session_info(self):
        """Display current session information."""
        session_info = self.session_detector.get_session_info()
        next_change = self.session_detector.get_next_session_change()
        
        print(f"\n🕐 Session Information:")
        print(f"   Current: {session_info['session']} ({session_info['phase']})")
        print(f"   UTC: {session_info['utc_time']}, IST: {session_info['ist_time']}")
        print(f"   Major Session: {session_info['is_major_session']}")
        print(f"   Next Change: {next_change['next_session']} in {next_change['time_until']}")


def run_example():
    """Run the complete example."""
    print("🚀 Order Database Integration Example")
    print("=" * 50)
    
    example = OrderIntegrationExample()
    
    # Show current session
    example.show_session_info()
    
    # Create sample order
    example.create_sample_order("SMC_STRATEGY")
    
    # Simulate order lifecycle
    example.simulate_order_lifecycle()
    
    # Show statistics
    example.show_statistics()
    
    # Export data
    example.export_data()
    
    print(f"\n✅ Example completed successfully!")
    print(f"💡 Access your data via API at: http://localhost:8899/orders/")


if __name__ == "__main__":
    run_example()
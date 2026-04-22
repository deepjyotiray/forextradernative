"""
PnL Validator - Enhanced accuracy validation for order PnL data.
Cross-references multiple MT5 sources to ensure accurate PnL tracking.
"""
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import config as cfg


class PnLValidator:
    """Validates and ensures accurate PnL data from MT5."""
    
    def __init__(self, mt5_bridge):
        self.bridge = mt5_bridge
        self._pnl_cache = {}  # Cache for validation
    
    def get_accurate_live_pnl(self, ticket: int) -> Dict:
        """
        Get highly accurate live PnL with validation.
        
        Returns:
            {
                'pnl': float,
                'breakdown': {'profit': float, 'swap': float, 'commission': float},
                'confidence': str,  # 'HIGH', 'MEDIUM', 'LOW'
                'source': str,
                'validation_notes': List[str]
            }
        """
        validation_notes = []
        
        # Method 1: Direct position query
        pos_pnl = self._get_position_pnl(ticket)
        
        # Method 2: Deal history aggregation (for validation)
        deal_pnl = self._get_deal_history_pnl(ticket)
        
        # Method 3: Order history (fallback)
        order_pnl = self._get_order_history_pnl(ticket)
        
        # Determine best source and confidence
        result = self._validate_pnl_sources(pos_pnl, deal_pnl, order_pnl, validation_notes)
        result['validation_notes'] = validation_notes
        
        return result
    
    def _get_position_pnl(self, ticket: int) -> Optional[Dict]:
        """Get PnL from current position."""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return None
            
            pos = positions[0]
            return {
                'pnl': round(pos.profit + pos.swap + getattr(pos, 'commission', 0.0), 2),
                'breakdown': {
                    'profit': round(pos.profit, 2),
                    'swap': round(pos.swap, 2),
                    'commission': round(getattr(pos, 'commission', 0.0), 2)
                },
                'volume': pos.volume,
                'current_price': pos.price_current,
                'source': 'position'
            }
        except Exception as e:
            return None
    
    def _get_deal_history_pnl(self, ticket: int) -> Optional[Dict]:
        """Get PnL from deal history aggregation."""
        try:
            now = datetime.now(timezone.utc)
            deals = mt5.history_deals_get(now - timedelta(days=7), now + timedelta(hours=1))
            
            if not deals:
                return None
            
            total_profit = 0.0
            total_swap = 0.0
            total_commission = 0.0
            deal_count = 0
            
            for deal in deals:
                if deal.position_id == ticket:
                    total_profit += deal.profit
                    total_swap += deal.swap
                    total_commission += deal.commission
                    deal_count += 1
            
            if deal_count == 0:
                return None
            
            return {
                'pnl': round(total_profit + total_swap + total_commission, 2),
                'breakdown': {
                    'profit': round(total_profit, 2),
                    'swap': round(total_swap, 2),
                    'commission': round(total_commission, 2)
                },
                'deal_count': deal_count,
                'source': 'deals'
            }
        except Exception:
            return None
    
    def _get_order_history_pnl(self, ticket: int) -> Optional[Dict]:
        """Get PnL from order history (fallback method)."""
        try:
            now = datetime.now(timezone.utc)
            orders = mt5.history_orders_get(now - timedelta(days=7), now + timedelta(hours=1))
            
            if not orders:
                return None
            
            position_orders = [o for o in orders if o.position_id == ticket and o.state == 1]
            
            if len(position_orders) < 2:  # Need at least open + close
                return None
            
            position_orders.sort(key=lambda o: o.time_setup)
            open_order = position_orders[0]
            close_order = position_orders[-1]
            
            # Calculate PnL from price difference
            direction = "BUY" if open_order.type == 0 else "SELL"
            entry_price = open_order.price_current or open_order.price_open
            exit_price = close_order.price_current or close_order.price_open
            volume = open_order.volume_initial
            
            if direction == "BUY":
                price_pnl = (exit_price - entry_price) * volume * cfg.PIP_VALUE_PER_LOT
            else:
                price_pnl = (entry_price - exit_price) * volume * cfg.PIP_VALUE_PER_LOT
            
            return {
                'pnl': round(price_pnl, 2),
                'breakdown': {
                    'profit': round(price_pnl, 2),
                    'swap': 0.0,  # Not available from orders
                    'commission': 0.0  # Not available from orders
                },
                'entry_price': entry_price,
                'exit_price': exit_price,
                'source': 'orders'
            }
        except Exception:
            return None
    
    def _validate_pnl_sources(self, pos_pnl: Dict, deal_pnl: Dict, order_pnl: Dict, 
                             notes: List[str]) -> Dict:
        """Validate and choose best PnL source."""
        
        # Priority: Position > Deals > Orders
        if pos_pnl:
            confidence = "HIGH"
            primary_source = pos_pnl
            notes.append("Using live position data (highest accuracy)")
            
            # Cross-validate with deals if available
            if deal_pnl:
                diff = abs(pos_pnl['pnl'] - deal_pnl['pnl'])
                if diff > 0.50:  # Significant difference
                    confidence = "MEDIUM"
                    notes.append(f"Position/Deal PnL difference: ${diff:.2f}")
                else:
                    notes.append("Position data validated against deal history")
        
        elif deal_pnl:
            confidence = "MEDIUM"
            primary_source = deal_pnl
            notes.append("Using deal history aggregation")
            
            # Check if position is actually closed
            if deal_pnl['deal_count'] > 1:
                notes.append("Multiple deals found - position likely closed")
        
        elif order_pnl:
            confidence = "LOW"
            primary_source = order_pnl
            notes.append("Using order history calculation (swap/commission not included)")
        
        else:
            return {
                'pnl': 0.0,
                'breakdown': {'profit': 0.0, 'swap': 0.0, 'commission': 0.0},
                'confidence': 'NONE',
                'source': 'none',
                'validation_notes': ['No PnL data available from any source']
            }
        
        return {
            'pnl': primary_source['pnl'],
            'breakdown': primary_source['breakdown'],
            'confidence': confidence,
            'source': primary_source['source']
        }
    
    def validate_closed_position_pnl(self, ticket: int) -> Dict:
        """
        Comprehensive validation for closed position PnL.
        Uses multiple sources to ensure accuracy.
        """
        validation_notes = []
        
        # Get deal history (most accurate for closed positions)
        deal_pnl = self._get_deal_history_pnl(ticket)
        
        # Get order history (for cross-validation)
        order_pnl = self._get_order_history_pnl(ticket)
        
        if deal_pnl:
            confidence = "HIGH"
            final_pnl = deal_pnl['pnl']
            breakdown = deal_pnl['breakdown']
            notes.append(f"Deal history: {deal_pnl['deal_count']} deals processed")
            
            # Cross-validate with order calculation
            if order_pnl:
                diff = abs(deal_pnl['breakdown']['profit'] - order_pnl['breakdown']['profit'])
                if diff > 1.0:  # Allow for swap/commission differences
                    notes.append(f"Deal/Order profit difference: ${diff:.2f}")
                else:
                    notes.append("Deal data validated against order history")
        
        elif order_pnl:
            confidence = "MEDIUM"
            final_pnl = order_pnl['pnl']
            breakdown = order_pnl['breakdown']
            notes.append("Using order history (swap/commission estimated as 0)")
        
        else:
            confidence = "NONE"
            final_pnl = 0.0
            breakdown = {'profit': 0.0, 'swap': 0.0, 'commission': 0.0}
            notes.append("No reliable PnL data found")
        
        return {
            'final_pnl': final_pnl,
            'breakdown': breakdown,
            'confidence': confidence,
            'validation_notes': notes,
            'validated_at': datetime.now(timezone.utc).isoformat()
        }
    
    def get_account_pnl_summary(self) -> Dict:
        """Get comprehensive account PnL summary with validation."""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return {'error': 'Account info not available'}
            
            # Get all positions
            positions = mt5.positions_get()
            floating_pnl = sum(p.profit + p.swap + getattr(p, 'commission', 0.0) 
                             for p in (positions or []))
            
            # Get today's closed PnL
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
            deals = mt5.history_deals_get(today_start, datetime.now(timezone.utc))
            
            today_pnl = 0.0
            if deals:
                for deal in deals:
                    if deal.magic == cfg.MAGIC_NUMBER:
                        today_pnl += deal.profit + deal.swap + deal.commission
            
            return {
                'account_balance': account_info.balance,
                'account_equity': account_info.equity,
                'floating_pnl': round(floating_pnl, 2),
                'today_realized_pnl': round(today_pnl, 2),
                'total_positions': len(positions or []),
                'validation_time': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
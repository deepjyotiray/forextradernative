"""
Trade Attribution Engine - Comprehensive logging for every trade decision.

Logs for every decision (taken or skipped):
- setup.direction, bias.direction, trade_type
- quality_score, threshold_used
- spread metrics, ATR, compression/LTF flags
- tick metrics, session

For executed trades, also logs:
- entry/exit prices, SL/TP, outcome, PnL
- trade_duration, exit_reason
"""
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional, List
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ATTRIBUTION_FILE = os.path.join(_BASE_DIR, "trade_attribution.jsonl")


class TradeAttributionEngine:
    def __init__(self):
        self._active_trades = {}  # {trade_id: attribution_data}
        self._trade_counter = 0
    
    def log_decision(
        self,
        strategy: str,
        setup_direction: Optional[str],
        bias_direction: Optional[str],
        quality_score: float,
        threshold_used: float,
        spread_mean: float,
        spread_std: float,
        spread_percentile: float,
        atr_value: float,
        compression_flag: bool,
        ltf_conflict_flag: bool,
        tick_ratio: float,
        tick_velocity: float,
        session: str,
        decision: str,  # "TRADE_TAKEN" or "TRADE_SKIPPED"
        reason: str,
        price: float,
        additional_data: Optional[Dict] = None
    ) -> Optional[str]:
        """Log a trade decision. Returns trade_id if trade was taken."""
        
        # Determine trade type
        trade_type = "NEUTRAL"
        if setup_direction and bias_direction and bias_direction != "NEUTRAL":
            if setup_direction == bias_direction:
                trade_type = "WITH_TREND"
            else:
                trade_type = "COUNTER_TREND"
        
        attribution = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "unix_time": time.time(),
            "strategy": strategy,
            "setup_direction": setup_direction,
            "bias_direction": bias_direction,
            "trade_type": trade_type,
            "quality_score": round(quality_score, 3),
            "threshold_used": round(threshold_used, 3),
            "spread_mean": round(spread_mean, 4),
            "spread_std": round(spread_std, 4),
            "spread_percentile": round(spread_percentile, 3),
            "atr_value": round(atr_value, 4),
            "compression_flag": compression_flag,
            "ltf_conflict_flag": ltf_conflict_flag,
            "tick_ratio": round(tick_ratio, 3),
            "tick_velocity": round(tick_velocity, 2),
            "session": session,
            "decision": decision,
            "reason": reason,
            "price": round(price, 2),
        }
        
        if additional_data:
            attribution.update(additional_data)
        
        trade_id = None
        if decision == "TRADE_TAKEN":
            self._trade_counter += 1
            trade_id = f"{strategy}_{int(time.time())}_{self._trade_counter}"
            attribution["trade_id"] = trade_id
            self._active_trades[trade_id] = attribution
        
        self._write_attribution(attribution)
        return trade_id
    
    def log_trade_outcome(
        self,
        trade_id: str,
        entry_price: float,
        exit_price: float,
        sl: float,
        tp: float,
        outcome: str,  # "WIN", "LOSS", "BE"
        pnl: float,
        trade_duration: int,  # seconds
        exit_reason: str  # "TP", "SL", "TIMEOUT", "EARLY_EXIT"
    ):
        """Log the outcome of an executed trade."""
        
        if trade_id not in self._active_trades:
            return
        
        attribution = self._active_trades[trade_id].copy()
        attribution.update({
            "trade_completed": True,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "outcome": outcome,
            "pnl": round(pnl, 2),
            "trade_duration": trade_duration,
            "exit_reason": exit_reason,
            "completion_time": datetime.now(timezone.utc).isoformat(),
        })
        
        self._write_attribution(attribution)
        del self._active_trades[trade_id]
    
    def _write_attribution(self, attribution: Dict):
        """Write attribution data to file."""
        try:
            with open(_ATTRIBUTION_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(attribution) + "\n")
        except Exception as e:
            print(f"[ATTRIBUTION ERROR] Failed to write: {e}")
    
    def get_recent_attributions(self, hours: int = 24) -> List[Dict]:
        """Get recent attribution data."""
        if not os.path.exists(_ATTRIBUTION_FILE):
            return []
        
        cutoff = time.time() - (hours * 3600)
        attributions = []
        
        try:
            with open(_ATTRIBUTION_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        attr = json.loads(line.strip())
                        if attr.get("unix_time", 0) >= cutoff:
                            attributions.append(attr)
                    except:
                        continue
        except Exception:
            pass
        
        return attributions
    
    def get_active_trades(self) -> Dict:
        """Get currently active trades."""
        return self._active_trades.copy()


# Singleton instance
attribution_engine = TradeAttributionEngine()


def log_trade_decision(
    strategy: str,
    setup_direction: Optional[str],
    bias_direction: Optional[str],
    quality_score: float,
    threshold_used: float,
    spread_data: Dict,
    atr_value: float,
    compression_flag: bool,
    ltf_conflict_flag: bool,
    tick_data: Dict,
    session: str,
    decision: str,
    reason: str,
    price: float,
    additional_data: Optional[Dict] = None
) -> Optional[str]:
    """Convenience function to log trade decisions."""
    
    return attribution_engine.log_decision(
        strategy=strategy,
        setup_direction=setup_direction,
        bias_direction=bias_direction,
        quality_score=quality_score,
        threshold_used=threshold_used,
        spread_mean=spread_data.get("mean", 0),
        spread_std=spread_data.get("std", 0),
        spread_percentile=spread_data.get("percentile", 0),
        atr_value=atr_value,
        compression_flag=compression_flag,
        ltf_conflict_flag=ltf_conflict_flag,
        tick_ratio=tick_data.get("ratio", 0.5),
        tick_velocity=tick_data.get("velocity", 0),
        session=session,
        decision=decision,
        reason=reason,
        price=price,
        additional_data=additional_data
    )


def log_trade_outcome(
    trade_id: str,
    entry_price: float,
    exit_price: float,
    sl: float,
    tp: float,
    outcome: str,
    pnl: float,
    trade_duration: int,
    exit_reason: str
):
    """Convenience function to log trade outcomes."""
    
    attribution_engine.log_trade_outcome(
        trade_id=trade_id,
        entry_price=entry_price,
        exit_price=exit_price,
        sl=sl,
        tp=tp,
        outcome=outcome,
        pnl=pnl,
        trade_duration=trade_duration,
        exit_reason=exit_reason
    )


def get_recent_attributions(hours: int = 24) -> List[Dict]:
    """Get recent attribution data."""
    return attribution_engine.get_recent_attributions(hours)


def get_active_trades() -> Dict:
    """Get currently active trades."""
    return attribution_engine.get_active_trades()
"""
Master Control System - Coordinates all measurement, validation, and risk protection.

Integrates:
- Trade Attribution Engine
- Performance Analytics
- Parameter Calibration
- Over-filtering Detection
- Session Risk Control
- Trade Pacing
- Quality Feedback Loop
- Reporting Dashboard
"""
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from .session_filter import get_session, is_market_open
from .trade_attribution import log_trade_decision, log_trade_outcome
from .session_risk_control import check_trade_allowed, record_trade_outcome
from .trade_pacing import check_pacing_allowed
from .over_filtering_detection import get_active_relaxations
from .parameter_calibration import get_current_parameters
from .trade_quality_feedback import check_analysis_needed, perform_quality_analysis
from .xgb_model import apply_xgb_filter


class MasterControlSystem:
    def __init__(self):
        self._active_trades = {}  # {trade_id: trade_info}
        self._system_status = "active"
        
    def pre_trade_validation(self, signal: Dict, strategy: str, market_data: Dict) -> Tuple[bool, Dict]:
        """Comprehensive pre-trade validation."""
        
        validation_result = {
            "allowed": True,
            "blocks": [],
            "warnings": [],
            "system_status": {}
        }
        
        # 1. Session Risk Control Check
        risk_check = check_trade_allowed()
        if not risk_check["allowed"]:
            validation_result["allowed"] = False
            validation_result["blocks"].append({
                "type": "risk_control",
                "reason": risk_check["reason"]
            })
        
        validation_result["system_status"]["risk_control"] = risk_check["risk_status"]
        
        # 2. Trade Pacing Check
        pacing_check = check_pacing_allowed()
        if not pacing_check["allowed"]:
            validation_result["allowed"] = False
            validation_result["blocks"].append({
                "type": "trade_pacing",
                "reason": pacing_check["reason"]
            })
        
        validation_result["system_status"]["trade_pacing"] = pacing_check
        
        # 3. Apply Parameter Calibration
        current_params = get_current_parameters()
        validation_result["system_status"]["parameters"] = current_params
        
        # 4. Apply Over-filtering Relaxations
        active_relaxations = get_active_relaxations()
        validation_result["system_status"]["filter_relaxations"] = active_relaxations
        
        # 5. XGBoost Filter (if signal is still allowed)
        if validation_result["allowed"] and signal.get("signal") in ("BUY", "SELL"):
            indicators = market_data.get("indicators") or {}
            regime = market_data.get("regime") or {}
            bias = market_data.get("bias") or {}
            tick = market_data.get("tick") or {}
            
            filtered_signal = apply_xgb_filter(signal, indicators, regime, bias, tick)
            
            if filtered_signal.get("signal") == "NO_TRADE":
                validation_result["allowed"] = False
                validation_result["blocks"].append({
                    "type": "xgboost_filter",
                    "reason": filtered_signal.get("reason", "XGBoost filter blocked trade")
                })
            else:
                # Update signal with XGB info
                signal.update(filtered_signal)
        
        return validation_result["allowed"], validation_result
    
    def log_trade_decision_comprehensive(
        self,
        strategy: str,
        signal: Dict,
        market_data: Dict,
        decision: str,  # "TRADE_TAKEN" or "TRADE_SKIPPED"
        reason: str,
        validation_result: Optional[Dict] = None
    ) -> Optional[str]:
        """Log comprehensive trade decision with all context."""
        validation_result = validation_result or {}
        
        # Extract data for attribution
        setup_direction = signal.get("_setup_direction")
        bias_direction = signal.get("_bias_direction")
        quality_score = signal.get("_quality_score", signal.get("confidence", 0))
        threshold_used = signal.get("_threshold", 0.65)
        
        # Market data
        tick_data = market_data.get("tick") or {}
        indicators = market_data.get("indicators") or {}
        
        # Spread data
        spread_data = {
            "mean": tick_data.get("spread_mean", tick_data.get("spread", 0)),
            "std": tick_data.get("spread_std", 0),
            "percentile": tick_data.get("spread_pctl", tick_data.get("spread_percentile", 0.5))
        }
        
        # Tick data
        tick_metrics = {
            "ratio": tick_data.get("dir_pct", 0.5),
            "velocity": tick_data.get("velocity", 0)
        }
        
        # Flags
        compression_flag = not any(b.get("type") == "compression" for b in validation_result.get("blocks", []))
        ltf_conflict_flag = signal.get("_ltf_conflict", False)
        
        # Session
        session = self._get_current_session()
        
        # Price
        price = signal.get("entry", (market_data.get("tick") or {}).get("bid", 0))
        
        # Additional data
        additional_data = {
            "strategy_specific": {
                "sl": signal.get("sl"),
                "tp": signal.get("tp"),
                "rr": signal.get("rr"),
                "counter_trend": signal.get("_counter_trend", False)
            },
            "validation_result": validation_result,
            "market_context": {
                "atr": indicators.get("atr14", indicators.get("atr", 0)),
                "body_ratio": indicators.get("body_ratio", 0),
                "regime": (market_data.get("regime") or {}).get("state"),
            }
        }
        
        # Log the decision
        trade_id = log_trade_decision(
            strategy=strategy,
            setup_direction=setup_direction,
            bias_direction=bias_direction,
            quality_score=quality_score,
            threshold_used=threshold_used,
            spread_data=spread_data,
            atr_value=indicators.get("atr14", indicators.get("atr", 0)),
            compression_flag=compression_flag,
            ltf_conflict_flag=ltf_conflict_flag,
            tick_data=tick_metrics,
            session=session,
            decision=decision,
            reason=reason,
            price=price,
            additional_data=additional_data
        )
        
        # Store trade info for outcome logging (counters are updated in auto_trader.py after MT5 execution)
        if decision == "TRADE_TAKEN" and trade_id:
            self._active_trades[trade_id] = {
                "signal": signal.copy(),
                "start_time": time.time(),
                "strategy": strategy
            }

        return trade_id
    
    def log_trade_outcome_comprehensive(
        self,
        trade_id: str,
        exit_price: float,
        outcome: str,  # "WIN", "LOSS", "BE"
        exit_reason: str,  # "TP", "SL", "TIMEOUT", "EARLY_EXIT"
        pnl: float
    ):
        """Log comprehensive trade outcome."""
        
        if trade_id not in self._active_trades:
            return
        
        trade_info = self._active_trades[trade_id]
        signal = trade_info["signal"]
        start_time = trade_info["start_time"]
        
        # Calculate trade duration
        trade_duration = int(time.time() - start_time)
        
        # Log to attribution engine
        log_trade_outcome(
            trade_id=trade_id,
            entry_price=signal.get("entry", 0),
            exit_price=exit_price,
            sl=signal.get("sl", 0),
            tp=signal.get("tp", 0),
            outcome=outcome,
            pnl=pnl,
            trade_duration=trade_duration,
            exit_reason=exit_reason
        )
        
        # Update risk control
        record_trade_outcome(pnl, outcome)
        
        # Clean up
        del self._active_trades[trade_id]
        
        # Check if quality analysis is needed
        analysis_check = check_analysis_needed()
        if analysis_check["analysis_needed"]:
            # Trigger quality analysis (could be done asynchronously)
            perform_quality_analysis()
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        from .session_risk_control import get_risk_status
        from .trade_pacing import get_pacing_status
        from .over_filtering_detection import analyze_filtering_status
        from .parameter_calibration import get_current_parameters
        from .trade_quality_feedback import get_current_recommendations
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_active": self._system_status == "active",
            "active_trades": len(self._active_trades),
            "risk_control": get_risk_status(),
            "trade_pacing": get_pacing_status(),
            "filtering_status": analyze_filtering_status(),
            "current_parameters": get_current_parameters(),
            "current_recommendations": get_current_recommendations(),
            "quality_analysis": check_analysis_needed(),
        }
    
    def emergency_stop(self, reason: str) -> Dict:
        """Emergency stop all trading."""
        self._system_status = "stopped"
        
        return {
            "status": "emergency_stop_activated",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_trades_count": len(self._active_trades)
        }
    
    def resume_trading(self) -> Dict:
        """Resume trading after emergency stop."""
        self._system_status = "active"
        
        return {
            "status": "trading_resumed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_current_session(self) -> str:
        """Get current trading session."""
        return get_session() if is_market_open() else "CLOSED"


# Singleton instance
master_control = MasterControlSystem()


def pre_trade_validation(signal: Dict, strategy: str, market_data: Dict) -> Tuple[bool, Dict]:
    """Comprehensive pre-trade validation."""
    return master_control.pre_trade_validation(signal, strategy, market_data)


def log_trade_decision_comprehensive(
    strategy: str,
    signal: Dict,
    market_data: Dict,
    decision: str,
    reason: str,
    validation_result: Optional[Dict] = None
) -> Optional[str]:
    """Log comprehensive trade decision."""
    return master_control.log_trade_decision_comprehensive(
        strategy, signal, market_data, decision, reason, validation_result
    )


def log_trade_outcome_comprehensive(
    trade_id: str,
    exit_price: float,
    outcome: str,
    exit_reason: str,
    pnl: float
):
    """Log comprehensive trade outcome."""
    master_control.log_trade_outcome_comprehensive(
        trade_id, exit_price, outcome, exit_reason, pnl
    )


def get_system_status() -> Dict:
    """Get comprehensive system status."""
    return master_control.get_system_status()


def emergency_stop(reason: str) -> Dict:
    """Emergency stop trading."""
    return master_control.emergency_stop(reason)


def resume_trading() -> Dict:
    """Resume trading."""
    return master_control.resume_trading()

"""
Session Detector - Automatically identify trading sessions and market phases.
Used to enrich order data with session context.
"""
from datetime import datetime, timezone, timedelta
from typing import Tuple
import config as cfg

_IST = timezone(timedelta(hours=5, minutes=30))


class SessionDetector:
    """Detect current trading session and market phase."""
    
    @staticmethod
    def get_current_session() -> Tuple[str, str]:
        """
        Get current trading session and market phase.
        
        Returns:
            Tuple[session_type, market_phase]
            
        Sessions (UTC):
        - ASIAN: 00:00-07:00
        - LONDON: 07:00-13:00  
        - NY: 13:00-20:00
        - QUIET: 20:00-00:00
        
        Market Phases:
        - OPEN: First 30 minutes of session
        - ACTIVE: Main trading hours
        - CLOSE: Last 30 minutes of session
        - OVERLAP: Session overlap periods
        - QUIET: Low activity periods
        """
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        minute = now_utc.minute
        
        # Determine session
        if 0 <= hour < 7:
            session = "ASIAN"
            session_start = 0
            session_end = 7
        elif 7 <= hour < 13:
            session = "LONDON"
            session_start = 7
            session_end = 13
        elif 13 <= hour < 20:
            session = "NY"
            session_start = 13
            session_end = 20
        else:  # 20-24
            session = "QUIET"
            session_start = 20
            session_end = 24
        
        # Determine market phase
        current_minutes = hour * 60 + minute
        session_start_minutes = session_start * 60
        session_end_minutes = session_end * 60
        
        # Handle session overlaps
        if hour == 7:  # London open (Asian close)
            phase = "OVERLAP" if minute < 30 else "OPEN"
        elif hour == 13:  # NY open (London close)
            phase = "OVERLAP" if minute < 30 else "OPEN"
        elif session == "QUIET":
            phase = "QUIET"
        else:
            # Within session - check if open/close/active
            if current_minutes <= session_start_minutes + 30:
                phase = "OPEN"
            elif current_minutes >= session_end_minutes - 30:
                phase = "CLOSE"
            else:
                phase = "ACTIVE"
        
        return session, phase
    
    @staticmethod
    def get_session_info() -> dict:
        """Get detailed session information."""
        session, phase = SessionDetector.get_current_session()
        now_utc = datetime.now(timezone.utc)
        now_ist = now_utc.astimezone(_IST)
        
        return {
            "session": session,
            "phase": phase,
            "utc_time": now_utc.strftime("%H:%M:%S"),
            "ist_time": now_ist.strftime("%H:%M:%S"),
            "is_major_session": session in ["LONDON", "NY"],
            "is_overlap": phase == "OVERLAP",
            "is_quiet_time": session == "QUIET" or phase == "QUIET"
        }
    
    @staticmethod
    def is_trading_hours() -> bool:
        """Check if current time is within active trading hours."""
        session, phase = SessionDetector.get_current_session()
        return session != "QUIET" and phase != "QUIET"
    
    @staticmethod
    def get_next_session_change() -> dict:
        """Get information about the next session change."""
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        
        # Define session boundaries
        boundaries = [
            (0, "ASIAN_START"),
            (7, "LONDON_START"), 
            (13, "NY_START"),
            (20, "QUIET_START")
        ]
        
        # Find next boundary
        next_boundary = None
        for boundary_hour, boundary_name in boundaries:
            if hour < boundary_hour:
                next_boundary = (boundary_hour, boundary_name)
                break
        
        # If no boundary found today, use tomorrow's first boundary
        if not next_boundary:
            next_boundary = (24, "ASIAN_START")  # Tomorrow's Asian start
        
        next_hour, next_name = next_boundary
        
        # Calculate time until next session
        if next_hour == 24:
            next_time = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_time = now_utc.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        
        time_until = next_time - now_utc
        minutes_until = int(time_until.total_seconds() / 60)
        
        return {
            "next_session": next_name,
            "next_time_utc": next_time.strftime("%H:%M:%S"),
            "minutes_until": minutes_until,
            "time_until": f"{minutes_until // 60}h {minutes_until % 60}m"
        }
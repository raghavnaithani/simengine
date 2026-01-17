from datetime import datetime
from pathlib import Path
import os
import json
from typing import Any, Optional

# Separate log files
PROJECT_LOG_PATH = Path(os.getcwd()).resolve().parents[0] / 'project_log.txt'
ERROR_LOG_PATH = Path(os.getcwd()).resolve().parents[0] / 'error_log.txt'


def _ensure_log_path(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)


def record_event(level: str = 'INFO', action: Optional[str] = None, message: Optional[str] = None, details: Optional[Any] = None) -> None:
    """Write a structured event to error_log.txt (for raw backend logs).
    
    NOTE: This is for backend system logs. Agent observations go to project_log.txt via agent_log().
    """
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    action_part = f" {action}" if action else ""
    msg_part = message or ""

    human_line = f"[{ts}] [{level}]{action_part} {msg_part}\n"

    # Structured event as JSON (single-line)
    event = {
        'timestamp': ts,
        'level': level,
        'action': action,
        'message': message,
        'details': details,
    }

    _ensure_log_path(ERROR_LOG_PATH)
    with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(human_line)
        if details is not None:
            try:
                j = json.dumps(event, ensure_ascii=False)
            except Exception:
                # Fallback: stringify details safely
                event['details'] = str(details)
                j = json.dumps(event, ensure_ascii=False)
            f.write(j + "\n")


def append_log(message: str, level: str = 'INFO') -> None:
    """Backward-compatible wrapper - writes to error_log.txt for raw backend logs."""
    record_event(level=level, message=message)


def agent_log(message: str, level: str = 'INFO') -> None:
    """Write agent observations to project_log.txt (high-level summaries only, no raw data)."""
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{ts}] [{level}] {message}\n"
    
    _ensure_log_path(PROJECT_LOG_PATH)
    with open(PROJECT_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(log_line)
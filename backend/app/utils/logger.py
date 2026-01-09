from datetime import datetime
from pathlib import Path
import os

LOG_PATH = Path(os.getcwd()).resolve().parents[0] / 'project_log.txt'

def append_log(message: str, level: str = 'INFO') -> None:
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] [{level}] {message}\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line)

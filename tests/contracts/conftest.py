from __future__ import annotations

import sys
from pathlib import Path

CONTRACTS_DIR = Path(__file__).resolve().parent
if str(CONTRACTS_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACTS_DIR))

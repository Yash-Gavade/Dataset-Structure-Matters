from __future__ import annotations
import re
from typing import List

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\u0000-\u001f\u007f]+")

def normalize_text(s: str) -> str:
    '''Deterministic normalization for structural stats.'''
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CTRL_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()

def whitespace_tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return [] if not s else s.split(" ")

def safe_lower(s: str) -> str:
    return normalize_text(s).lower()

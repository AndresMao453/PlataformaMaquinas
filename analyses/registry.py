# analyses/registry.py
from __future__ import annotations
from typing import Callable, Dict, Any

_ANALYSES: Dict[str, Dict[str, Any]] = {}

def register_analysis(key: str, title: str, fn: Callable[..., Any]) -> None:
    _ANALYSES[key] = {"title": title, "fn": fn}

def get_analyses() -> Dict[str, Dict[str, Any]]:
    return _ANALYSES

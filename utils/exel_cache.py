# utils/excel_cache.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
import pandas as pd


CACHE_TTL_SEC = 45 * 60        # 45 min
MAX_FILES_IN_RAM = 10          # evita reventar memoria (ajústalo)

_lock = threading.RLock()
_store: dict[str, "_Entry"] = {}


@dataclass
class _Entry:
    path: Path
    mtime: float
    last_access: float
    xl: pd.ExcelFile
    sheets: dict[tuple, pd.DataFrame] = field(default_factory=dict)


def _purge(now: float) -> None:
    # TTL
    expired = [k for k, e in _store.items() if (now - e.last_access) > CACHE_TTL_SEC]
    for k in expired:
        try:
            _store[k].xl.close()
        except Exception:
            pass
        _store.pop(k, None)

    # LRU (si te pasas del máximo)
    if len(_store) > MAX_FILES_IN_RAM:
        items = sorted(_store.items(), key=lambda kv: kv[1].last_access)  # más viejo primero
        to_remove = len(_store) - MAX_FILES_IN_RAM
        for k, e in items[:to_remove]:
            try:
                e.xl.close()
            except Exception:
                pass
            _store.pop(k, None)


def get_excel_entry(filename_key: str, upload_dir: Path) -> _Entry:
    """
    filename_key: el nombre con el que guardaste el archivo (ej uuid_nombre.xlsx)
    """
    path = (upload_dir / filename_key).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Excel no existe: {path}")

    now = time.time()
    mtime = path.stat().st_mtime

    with _lock:
        _purge(now)

        e = _store.get(filename_key)
        # hit válido: mismo archivo (mtime igual)
        if e and e.mtime == mtime:
            e.last_access = now
            return e

        # miss o archivo cambió: recargar
        xl = pd.ExcelFile(path, engine="openpyxl")
        e = _Entry(path=path, mtime=mtime, last_access=now, xl=xl)
        _store[filename_key] = e
        return e


def get_sheet_df(filename_key: str, upload_dir: Path, sheet_name: str, usecols=None) -> pd.DataFrame:
    """
    Cachea por (sheet_name, usecols).
    OJO: si usecols cambia mucho, genera más entradas; mantén usecols estable.
    """
    e = get_excel_entry(filename_key, upload_dir)
    key = (sheet_name, tuple(usecols) if usecols is not None and not isinstance(usecols, str) else usecols)

    with _lock:
        if key in e.sheets:
            e.last_access = time.time()
            return e.sheets[key]

    df = e.xl.parse(sheet_name=sheet_name, usecols=usecols)

    with _lock:
        e.sheets[key] = df
        e.last_access = time.time()

    return df


def drop_file(filename_key: str) -> None:
    """Por si quieres liberar a mano cuando el usuario sube otro archivo."""
    with _lock:
        e = _store.pop(filename_key, None)
        if e:
            try:
                e.xl.close()
            except Exception:
                pass

import pandas as pd


def load_sheet(path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def df_preview_html(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None:
        df = pd.DataFrame()
    return df.head(max_rows).to_html(index=False, border=0)


# =========================
# Google Sheets (CSV por GID)
# =========================
import time
from typing import Dict, Tuple, Optional
import pandas as pd

_GSHEET_CACHE: Dict[Tuple[str, int], Tuple[float, pd.DataFrame]] = {}
_GSHEET_TTL_S_DEFAULT = 60

def _gsheet_csv_url(sheet_id: str, gid: int, cache_bust: int | None = None) -> str:
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    if cache_bust is None:
        return base
    return base + f"&_={int(cache_bust)}"


def load_gsheet_tab_csv(sheet_id: str, gid: int, ttl_s: Optional[int] = None) -> pd.DataFrame:
    ttl = _GSHEET_TTL_S_DEFAULT if ttl_s is None else int(ttl_s)
    key = (sheet_id, int(gid))
    now = time.time()

    hit = _GSHEET_CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1].copy()

    url = _gsheet_csv_url(sheet_id, int(gid), cache_bust=int(now))
    df = pd.read_csv(url)
    _GSHEET_CACHE[key] = (now, df.copy())
    return df


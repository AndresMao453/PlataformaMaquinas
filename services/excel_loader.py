import time
from typing import Dict, Tuple, Optional
import pandas as pd

_GSHEET_CACHE: Dict[Tuple[str, int], Tuple[float, pd.DataFrame]] = {}
_GSHEET_TTL_S_DEFAULT = 60

def _gsheet_csv_url(sheet_id: str, gid: int) -> str:
    # ✅ SIN cache_bust: deja que HTTP/CDN ayuden cuando corresponda
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={int(gid)}"

def load_gsheet_tab_csv(sheet_id: str, gid: int, ttl_s: Optional[int] = None) -> pd.DataFrame:
    ttl = _GSHEET_TTL_S_DEFAULT if ttl_s is None else int(ttl_s)
    key = (str(sheet_id), int(gid))
    now = time.time()

    hit = _GSHEET_CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        # ✅ NO copies gigantes: devolvemos el mismo DF
        return hit[1]

    url = _gsheet_csv_url(sheet_id, int(gid))
    df = pd.read_csv(url)

    # ✅ guardar una sola instancia
    _GSHEET_CACHE[key] = (now, df)
    return df

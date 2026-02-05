import re
import pandas as pd
from datetime import datetime, date

def _robust_to_datetime(s: pd.Series) -> pd.Series:
    """
    Convierte fechas de Excel / strings / datetime a datetime64[ns] de forma robusta.
    Soporta:
      - Excel serial (números tipo 45200)
      - "YYYY-MM-DD", "YYYY/MM/DD"
      - "DD/MM/YYYY" (si viene así)
      - datetime ya parseado
    """
    if s is None:
        return pd.to_datetime(pd.Series([], dtype="object"), errors="coerce")

    s2 = s.copy()

    # Si es numérico, intentar como serial de Excel (origen 1899-12-30)
    if pd.api.types.is_numeric_dtype(s2):
        # Excel serial típico: 20000..60000 (años ~1954..2064)
        return pd.to_datetime(s2, unit="D", origin="1899-12-30", errors="coerce")

    # Si es datetime ya
    if pd.api.types.is_datetime64_any_dtype(s2):
        return pd.to_datetime(s2, errors="coerce")

    # Si es string/object, parsear con flexibilidad
    # 1) intenta YYYY-MM-DD / YYYY/MM/DD
    dt = pd.to_datetime(s2.astype(str).str.strip(), errors="coerce", dayfirst=False)
    if dt.notna().any():
        return dt

    # 2) fallback dd/mm/yyyy
    return pd.to_datetime(s2.astype(str).str.strip(), errors="coerce", dayfirst=True)


def _pick_date_col(df: pd.DataFrame) -> str:
    """
    Intenta encontrar la columna de fecha.
    Si no encuentra nombres típicos, usa columna D (índice 3), como tú dijiste.
    """
    candidates = ["Fecha", "FECHA", "fecha", "Dia", "Día", "DATE", "Date"]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: columna D (4ta)
    if df.shape[1] >= 4:
        return df.columns[3]

    raise KeyError("No encontré columna de fecha (ni por nombre ni por columna D).")


def filter_by_period(df: pd.DataFrame, period: str, period_value: str, date_col: str | None = None) -> pd.DataFrame:
    """
    Filtra df por period/day-week-month usando la columna de fecha.
    Espera:
      - day: period_value = "YYYY-MM-DD"
      - month: "YYYY-MM" o "YYYY/MM"
      - week: "YYYY-W##" (ej: 2026-W02)
    """
    if df is None or df.empty:
        return df

    if not period_value:
        return df

    if date_col is None:
        date_col = _pick_date_col(df)

    dt = _robust_to_datetime(df[date_col])
    # normalizamos a fecha (sin hora)
    d = dt.dt.date

    p = (period or "").lower().strip()

    if p == "day":
        # acepta YYYY-MM-DD o YYYY/MM/DD
        pv = period_value.strip().replace("/", "-")
        target = datetime.strptime(pv, "%Y-%m-%d").date()
        out = df.loc[d == target].copy()
        return out

    if p == "month":
        pv = period_value.strip().replace("/", "-")
        # "YYYY-MM"
        m = datetime.strptime(pv, "%Y-%m").date()
        out = df.loc[(dt.dt.year == m.year) & (dt.dt.month == m.month)].copy()
        return out

    if p == "week":
        # "YYYY-W02"
        m = re.match(r"^\s*(\d{4})\s*-\s*W(\d{1,2})\s*$", period_value.strip(), flags=re.I)
        if not m:
            # fallback: si te llega otra cosa, no filtres (mejor que romper)
            return df
        y = int(m.group(1))
        w = int(m.group(2))
        iso = dt.dt.isocalendar()
        out = df.loc[(iso["year"] == y) & (iso["week"] == w)].copy()
        return out

    return df

# analyses/aplicacion_excel.py
from __future__ import annotations

import io
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from services.excel_loader import load_gsheet_tab_csv

GSHEET_APP_TOKEN = "__GSHEET_APP__"

# cache simple en memoria (para no bajar el XLSX en cada request)
# ✅ cache por sheet_id (soporta varias máquinas/sheets sin mezclar bytes)
_GS_CACHE: Dict[str, Dict[str, Any]] = {}  # {sheet_id: {"ts": float, "bytes": bytes}}
_GS_TTL_SEC = int(getattr(config, "GSHEET_TTL_S", 10) or 10)
# ✅ cache de DataFrames por (sheet_id, gid) para NO volver a descargar/parsear en cada endpoint
_GS_DF_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}  # {(sheet_id,gid): {"ts": float, "df": pd.DataFrame}}

def _load_gsheet_tab_csv_cached(sheet_id: str, gid: int, ttl_s: int) -> pd.DataFrame:
    """Carga un tab de Google Sheets con cache TTL (por sheet_id + gid)."""
    now = time.time()
    key = (str(sheet_id), int(gid))

    hit = _GS_DF_CACHE.get(key)
    if hit and (now - float(hit.get("ts", 0.0)) <= float(ttl_s)):
        return hit["df"]

    df = load_gsheet_tab_csv(str(sheet_id), int(gid), ttl_s=int(ttl_s))
    _GS_DF_CACHE[key] = {"ts": now, "df": df}
    return df



ANALYSIS_KEY = "aplicacion_excel"
ANALYSIS_TITLE = "Análisis Aplicación"


# ============================================================
# ✅ Selector Google Sheet según machine + machine_id
# ============================================================
def _pick_gsheet_params(machine: str, machine_id: str) -> Tuple[str, int, int, int]:
    """
    Devuelve (sheet_id, gid_resumen, gid_paradas, ttl_s) según máquina y machine_id.

    Mapeo:
    - APLICACION M1: config.GSHEET_ID  + GSHEET_GID_APLICACION  / GSHEET_GID_UNION
    - APLICACION M2: config.GSHEET_ID2 + GSHEET_GID_APLICACION2 / GSHEET_GID_UNION2
    - UNION M1:      config.GSHEET_ID3 + GSHEET_GID_APLICACION3 / GSHEET_GID_UNION3
    - UNION M2:      config.GSHEET_ID4 + GSHEET_GID_APLICACION4 / GSHEET_GID_UNION4
    """
    m = (machine or "APLICACION").strip().upper()
    mid = (machine_id or "").strip()

    ttl_s = int(getattr(config, "GSHEET_TTL_S", 60) or 60)

    # Defaults (Aplicación - Máquina 1)
    sheet_id = str(
        getattr(config, "GSHEET_ID", "")
        or getattr(config, "APLICACION_SHEET_ID", "")
        or ""
    ).strip()

    gid_res = int(
        getattr(config, "GSHEET_GID_APLICACION", 0)
        or getattr(config, "APLICACION_GID_APLICACION", 0)
        or 0
    )
    gid_par = int(
        getattr(config, "GSHEET_GID_UNION", 0)
        or getattr(config, "APLICACION_GID_UNION", 0)
        or 0
    )

    # Aplicación - Máquina 2
    if m == "APLICACION" and mid == "2":
        sheet_id = str(
            getattr(config, "GSHEET_ID2", sheet_id)
            or getattr(config, "APLICACION_SHEET_ID2", sheet_id)
            or sheet_id
        ).strip()
        gid_res = int(
            getattr(config, "GSHEET_GID_APLICACION2", gid_res)
            or getattr(config, "APLICACION_GID_APLICACION2", gid_res)
            or gid_res
        )
        gid_par = int(
            getattr(config, "GSHEET_GID_UNION2", gid_par)
            or getattr(config, "APLICACION_GID_UNION2", gid_par)
            or gid_par
        )

    # Unión - Máquina 1
    if m == "UNION" and mid == "1":
        sheet_id = str(getattr(config, "GSHEET_ID3", sheet_id) or sheet_id).strip()
        gid_res = int(getattr(config, "GSHEET_GID_APLICACION3", gid_res) or gid_res)
        gid_par = int(getattr(config, "GSHEET_GID_UNION3", gid_par) or gid_par)

    # ✅ Unión - Máquina 2 (NUEVO)
    if m == "UNION" and mid == "2":
        sheet_id = str(getattr(config, "GSHEET_ID4", sheet_id) or sheet_id).strip()
        gid_res = int(getattr(config, "GSHEET_GID_APLICACION4", gid_res) or gid_res)
        gid_par = int(getattr(config, "GSHEET_GID_UNION4", gid_par) or gid_par)

    if not sheet_id or not gid_res or not gid_par:
        raise RuntimeError("Faltan parámetros de Google Sheets (ID/GID) para la selección actual.")

    return sheet_id, gid_res, gid_par, ttl_s


def _norm_machine_id(machine_id: Any) -> str:
    return "" if machine_id is None else str(machine_id).strip()


# ============================================================
# Google Sheets XLSX (fallback / token)
# ============================================================
def _download_gsheet_xlsx_bytes(sheet_id: str) -> bytes:
    sheet_id = str(sheet_id or "").strip()
    if not sheet_id:
        raise RuntimeError("Falta sheet_id para descargar XLSX")

    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read()


def _get_gsheet_xlsx_bytes_cached(sheet_id: str) -> bytes:
    now = time.time()
    sid = str(sheet_id or "").strip()
    if not sid:
        raise RuntimeError("Falta sheet_id para cache XLSX")

    slot = _GS_CACHE.get(sid)
    if slot and slot.get("bytes") and (now - float(slot.get("ts", 0.0))) < _GS_TTL_SEC:
        return slot["bytes"]

    b = _download_gsheet_xlsx_bytes(sid)
    _GS_CACHE[sid] = {"ts": now, "bytes": b}
    return b


def _read_sheet_try(path: str, sheet_candidates: List[str], header=0, sheet_id: str = "") -> pd.DataFrame:
    """
    Intenta leer una de las hojas en sheet_candidates.
    Soporta:
      - path == GSHEET_APP_TOKEN  -> descarga XLSX del Sheet (sheet_id) y lee hoja
      - path == ruta a .xlsx      -> lee el Excel local
    """
    last = None
    for sh in sheet_candidates:
        try:
            if path == GSHEET_APP_TOKEN:
                if not sheet_id:
                    raise RuntimeError("Falta sheet_id para leer XLSX por token")
                xbytes = _get_gsheet_xlsx_bytes_cached(sheet_id)
                bio = io.BytesIO(xbytes)
                return pd.read_excel(bio, sheet_name=sh, header=header, engine="openpyxl")

            return pd.read_excel(path, sheet_name=sh, header=header, engine="openpyxl")

        except Exception as e:
            last = e
            continue

    raise last


# ============================================================
# Parse de hora robusto (sin warnings)
# ============================================================
_TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$")


def _parse_excel_time_series(s: pd.Series) -> pd.Series:
    """
    Convierte una columna de hora a datetime.time sin warnings:
    - Numérico Excel (fracción del día)
    - Strings HH:MM / HH:MM:SS (usa to_timedelta)
    - Strings con fecha+hora (fallback por elemento)
    """
    if s is None:
        return pd.Series([], dtype="object")

    if pd.api.types.is_numeric_dtype(s):
        td = pd.to_timedelta(s, unit="D", errors="coerce")
        base = pd.Timestamp("1900-01-01")
        return (base + td).dt.time

    ss = s.astype(str).str.strip()

    mask_time = ss.apply(lambda x: bool(_TIME_RE.match(x)))
    out = pd.Series([None] * len(ss), index=ss.index, dtype="object")

    if mask_time.any():
        td = pd.to_timedelta(ss[mask_time], errors="coerce")
        base = pd.Timestamp("1900-01-01")
        out.loc[mask_time] = (base + td).dt.time

    mask_rest = out.isna()
    if mask_rest.any():

        def _one(x: str):
            x = (x or "").strip()
            if not x or x.lower() in ("nan", "none"):
                return None
            try:
                dt = pd.to_datetime(x, errors="coerce", dayfirst=True)
                if pd.isna(dt):
                    return None
                return dt.time()
            except Exception:
                return None

        out.loc[mask_rest] = ss.loc[mask_rest].map(_one)

    return out


# ============================================================
# Helpers de lectura (headers embebidos)
# ============================================================
def _find_header_row(raw: pd.DataFrame, required: set, max_scan: int = 15) -> int:
    max_i = min(max_scan, len(raw))
    for i in range(max_i):
        row = raw.iloc[i].astype(str).str.strip().tolist()
        if required.issubset(set(row)):
            return i
    return 0


# ============================================================
# Time helpers
# ============================================================
def _hhmm_to_seconds(hhmm: str) -> int:
    hhmm = (hhmm or "").strip()
    if not hhmm:
        return 0
    parts = hhmm.split(":")
    if len(parts) < 2:
        return 0
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        return hh * 3600 + mm * 60
    except Exception:
        return 0


def _timeobj_to_seconds(t: Any) -> Optional[int]:
    """Acepta datetime.time o string HH:MM(:SS)."""
    if t is None or (isinstance(t, float) and np.isnan(t)) or pd.isna(t):
        return None
    if hasattr(t, "hour") and hasattr(t, "minute"):
        try:
            return int(t.hour) * 3600 + int(t.minute) * 60 + int(getattr(t, "second", 0))
        except Exception:
            return None
    try:
        s = str(t).strip()
        if not s:
            return None
        td = pd.to_timedelta(s, errors="coerce")
        if pd.isna(td):
            return None
        return int(td.total_seconds())
    except Exception:
        return None


# ============================================================
# Cargar hojas APLICACIÓN / UNIÓN
# ============================================================
def load_resumen_pulsos(path: str, machine: str = "APLICACION", machine_id: str = "") -> pd.DataFrame:
    """
    - path vacío => lee por CSV usando GID según machine/machine_id
    - path == GSHEET_APP_TOKEN => lee XLSX export (sheet_id según machine/machine_id)
    - path Excel local => lee por nombres de hoja candidates
    """
    sheet_id, gid_res, _gid_par, ttl_s = _pick_gsheet_params(machine, machine_id)
    m = (machine or "APLICACION").strip().upper()
    mid = (machine_id or "").strip()

    if not path:
        gs = _load_gsheet_tab_csv_cached(sheet_id, int(gid_res), ttl_s=ttl_s)
        raw = pd.concat([pd.DataFrame([gs.columns.tolist()]), gs.reset_index(drop=True)], ignore_index=True)

    else:
        # ✅ IMPORTANTE: NO mezclar UNION/APLICACION en el mismo orden.
        # Elegir candidates según la máquina seleccionada.
        if m == "UNION":
            sheet_candidates = [
                "UNION2_Resumen_Pulsos" if mid == "2" else "UNION1_Resumen_Pulsos",
                "Union2_Resumen_Pulsos" if mid == "2" else "Union1_Resumen_Pulsos",
                "UNION_Resumen_Pulsos",
                "Resumen_Pulsos_UNION",
            ]
        else:  # APLICACION
            sheet_candidates = [
                "APLICACION2_Resumen_Pulsos" if mid == "2" else "APLICACION1_Resumen_Pulsos",
                "Aplicacion2_Resumen_Pulsos" if mid == "2" else "Aplicacion1_Resumen_Pulsos",
                "APLICACION_Resumen_Pulsos",
                "Resumen_Pulsos",
            ]

        token_sheet_id = sheet_id if path == GSHEET_APP_TOKEN else ""
        raw = _read_sheet_try(path, sheet_candidates, header=None, sheet_id=token_sheet_id)

    hi = _find_header_row(raw, {"Nombre", "Fecha", "Hora", "Pulsos"})
    header = raw.iloc[hi].astype(str).str.strip().tolist()
    df = raw.iloc[hi + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all").reset_index(drop=True)

    cols = [c for c in ["Nombre", "Fecha", "Hora", "Pulsos", "Terminal", "Aplicación"] if c in df.columns]
    df = df[cols].copy()

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)

    if "Hora" in df.columns:
        df["Hora"] = _parse_excel_time_series(df["Hora"])

    df["Pulsos"] = pd.to_numeric(df["Pulsos"], errors="coerce").fillna(0).astype(int)

    df["Nombre"] = df["Nombre"].astype(str)
    if "Terminal" in df.columns:
        df["Terminal"] = df["Terminal"].astype(str)
    if "Aplicación" in df.columns:
        df["Aplicación"] = df["Aplicación"].astype(str)

    return df


def load_paradas(path: str, machine: str = "APLICACION", machine_id: str = "") -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    - path vacío => lee por CSV usando GID según machine/machine_id
    - path == GSHEET_APP_TOKEN => lee XLSX export (sheet_id según machine/machine_id)
    - path Excel local => lee por nombres de hoja candidates
    """
    sheet_id, _gid_res, gid_par, ttl_s = _pick_gsheet_params(machine, machine_id)
    m = (machine or "APLICACION").strip().upper()
    mid = (machine_id or "").strip()

    if not path:
        gs = _load_gsheet_tab_csv_cached(sheet_id, int(gid_par), ttl_s=ttl_s)
        raw = pd.concat([pd.DataFrame([gs.columns.tolist()]), gs.reset_index(drop=True)], ignore_index=True)

    else:
        # ✅ Si existe una hoja genérica "Paradas", se intenta primero.
        # Luego se busca SOLO en la familia correspondiente (UNION o APLICACION)
        if m == "UNION":
            sheet_candidates = [
                "Paradas",
                "UNION2_Paradas" if mid == "2" else "UNION1_Paradas",
                "UNION_Paradas",
            ]
        else:  # APLICACION
            sheet_candidates = [
                "Paradas",
                "APLICACION2_Paradas" if mid == "2" else "APLICACION1_Paradas",
                "Aplicacion2_Paradas" if mid == "2" else "Aplicacion1_Paradas",
                "APLICACION_Paradas",
            ]

        token_sheet_id = sheet_id if path == GSHEET_APP_TOKEN else ""
        raw = _read_sheet_try(path, sheet_candidates, header=None, sheet_id=token_sheet_id)

    hi = _find_header_row(raw, {"Fecha", "Nombre", "Hora de Inicio"})
    header = raw.iloc[hi].astype(str).str.strip().tolist()

    df = raw.iloc[hi + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all").reset_index(drop=True)

    # --- extraer matriz de códigos (si existe) ---
    code_col = None
    desc_col = None
    first_row = df.iloc[0].astype(str).str.strip() if len(df) else pd.Series([], dtype=str)

    for c in df.columns:
        lc = str(c).strip().lower()
        if "codigo" in lc or "código" in lc:
            code_col = c
        if "descripcion" in lc or "descripción" in lc:
            desc_col = c

    reasons = {}
    if code_col and desc_col:
        tmp = df[[code_col, desc_col]].copy()
        tmp[code_col] = pd.to_numeric(tmp[code_col], errors="coerce")
        tmp = tmp.dropna(subset=[code_col])
        for _, r in tmp.iterrows():
            reasons[int(r[code_col])] = str(r[desc_col])

    return df, reasons


# ============================================================
# Periodos disponibles
# ============================================================
def _dates_union(res_df: pd.DataFrame, par_df: pd.DataFrame) -> List[pd.Timestamp]:
    """
    Devuelve lista ordenada de Timestamps normalizados.
    Robusto ante fechas como string y fuerza dayfirst=True (DD/MM).
    """
    def _norm(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty or "Fecha" not in df.columns:
            return pd.Series([], dtype="datetime64[ns]")

        s = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
        s = s.dropna()
        if s.empty:
            return pd.Series([], dtype="datetime64[ns]")
        return s.dt.normalize()

    d1 = _norm(res_df)
    d2 = _norm(par_df)

    all_dates = pd.concat([d1, d2], ignore_index=True).dropna()
    if all_dates.empty:
        return []

    # únicos + ordenados (ya como datetime)
    uniq = pd.to_datetime(all_dates.unique(), errors="coerce")
    uniq = uniq[~pd.isna(uniq)]
    return sorted(list(uniq))


def get_period_options_aplicacion(
    path: str,
    period: str,
    machine: str = "APLICACION",
    machine_id: str = "",
) -> List[Dict[str, str]]:
    res_df = load_resumen_pulsos(path, machine=machine, machine_id=machine_id)
    par_df, _ = load_paradas(path, machine=machine, machine_id=machine_id)

    dates = _dates_union(res_df, par_df)

    # ✅ NO usar "if not dates" (puede fallar según tipo)
    if dates is None or len(dates) == 0:
        return []

    dates = [pd.Timestamp(d).date() for d in dates]

    if period == "day":
        dates_sorted = sorted(dates, reverse=True)
        return [{"value": d.isoformat(), "label": d.strftime("%d/%m/%Y")} for d in dates_sorted]

    if period == "month":
        months = sorted(set((d.year, d.month) for d in dates), reverse=True)
        return [{"value": f"{y}-{m:02d}", "label": f"{y}-{m:02d}"} for (y, m) in months]

    if period == "week":
        weeks = sorted(set((d.isocalendar()[0], d.isocalendar()[1]) for d in dates), reverse=True)
        return [{"value": f"{y}-W{w:02d}", "label": f"{y} - Semana {w:02d}"} for (y, w) in weeks]

    return []



def _filter_period(df: pd.DataFrame, period: str, period_value: str, date_col: str = "Fecha") -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return df.iloc[0:0]

    # ✅ FIX: forzar dayfirst=True (DD/MM) para que NO se pierdan filas en paradas
    s = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    if period == "day":
        d0 = pd.to_datetime(period_value, errors="coerce")
        if pd.isna(d0):
            return df.iloc[0:0]
        day = pd.Timestamp(d0).date()
        return df[s.dt.date == day]

    if period == "month":
        m = re.match(r"^(\d{4})-(\d{2})$", str(period_value))
        if not m:
            return df.iloc[0:0]
        y = int(m.group(1))
        mo = int(m.group(2))
        return df[(s.dt.year == y) & (s.dt.month == mo)]

    if period == "week":
        m = re.match(r"^(\d{4})-W(\d{2})$", str(period_value))
        if not m:
            return df.iloc[0:0]
        y = int(m.group(1))
        w = int(m.group(2))

        iso = s.dt.isocalendar()
        mask = (iso["year"] == y) & (iso["week"] == w)
        return df[mask.fillna(False)]

    return df.iloc[0:0]


# ============================================================
# Formatos UI (ES)
# ============================================================
def _fmt_int_es(n: Any) -> str:
    try:
        n = int(round(float(n)))
    except Exception:
        n = 0
    return f"{n:,}".replace(",", ".")


def _fmt_pct_es(x: float) -> str:
    try:
        return f"{x*100:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return ""


def _fmt_hhmm(seconds: Any) -> str:
    try:
        seconds = int(round(float(seconds)))
    except Exception:
        seconds = 0
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h:02d}:{m:02d}"


# ============================================================
# Filtros para UI (operarias + min/max hora)
# ============================================================
def get_app_filter_options_aplicacion(
    path: str,
    period: str,
    period_value: str,
    machine: str = "APLICACION",
    subcat: str = "",
    machine_id: str = "",
) -> Dict[str, Any]:
    res_df = load_resumen_pulsos(path, machine=machine, machine_id=machine_id)
    res_f = _filter_period(res_df, period, period_value, "Fecha")

    operators: List[str] = []
    if "Nombre" in res_f.columns and not res_f.empty:
        operators = sorted(res_f["Nombre"].dropna().astype(str).str.strip().unique().tolist())

    min_time = "00:00"
    max_time = "23:59"
    if "Hora" in res_f.columns and not res_f.empty:
        secs = res_f["Hora"].apply(_timeobj_to_seconds).dropna()
        secs = pd.to_numeric(secs, errors="coerce").dropna()
        if not secs.empty:
            min_time = _fmt_hhmm(float(secs.min()))
            max_time = _fmt_hhmm(float(secs.max()))

    return {"operators": operators, "min_time": min_time, "max_time": max_time}


# ============================================================
# Pareto paradas (exportable)
# ============================================================
def build_pareto_paradas(par_f: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
    if par_f is None or par_f.empty:
        return {"total_s": 0.0, "items": []}

    df = par_f.copy()

    if "downtime_s" not in df.columns:
        return {"total_s": 0.0, "items": []}

    df["downtime_s"] = pd.to_numeric(df["downtime_s"], errors="coerce").fillna(0.0).astype(float)
    df = df[df["downtime_s"] > 0].copy()
    if df.empty:
        return {"total_s": 0.0, "items": []}

    if "CausaCode" not in df.columns:
        df["CausaCode"] = pd.to_numeric(df.get("Causa", np.nan), errors="coerce")
    if "Tipo de Intervalo" not in df.columns:
        df["Tipo de Intervalo"] = df.get("Tipo de Intervalo", "").astype(str)

    code = pd.to_numeric(df["CausaCode"], errors="coerce")
    interval = df["Tipo de Intervalo"].astype(str).str.strip()
    desc = df.get("CausaDesc", pd.Series([""] * len(df), index=df.index)).astype(str).str.strip()

    has_code = code.notna()

    code_int = code.fillna(-1).astype(int).astype(str)
    label_code = np.where(
        (desc.fillna("") != "") & (desc.fillna("").str.lower() != "nan"),
        (code_int + " - " + desc),
        code_int,
    )

    label_interval = interval.replace({"": "Sin intervalo", "nan": "Sin intervalo", "None": "Sin intervalo"})
    df["pareto_label"] = np.where(has_code, label_code, label_interval)

    g = df.groupby("pareto_label", dropna=False)["downtime_s"].sum().sort_values(ascending=False)

    total = float(g.sum()) if len(g) else 0.0
    if total <= 0:
        return {"total_s": 0.0, "items": []}

    g_top = g.head(top_n)
    rest = float(g.iloc[top_n:].sum()) if len(g) > top_n else 0.0
    if rest > 0:
        g_top = pd.concat([g_top, pd.Series({"Otros": rest})])

    pct = (g_top / total) * 100.0
    cum = pct.cumsum()

    items = []
    for label, sec in g_top.items():
        sec_f = float(sec)
        items.append(
            {
                "label": str(label),
                "seconds": sec_f,
                "hhmm": _fmt_hhmm(sec_f),
                "pct": float(pct.loc[label]),
                "cum_pct": float(cum.loc[label]),
            }
        )

    return {"total_s": total, "items": items}


# ============================================================
# KPIs Aplicación / Unión (día/semana/mes) + filtros operaria/hora
# ============================================================
def run_aplicacion_analysis(
    path: str,
    period: str,
    period_value: str,
    machine: str = "APLICACION",
    subcat: Optional[str] = None,
    operator: str = "",
    time_start: str = "",
    time_end: str = "",
    machine_id: str = "",
) -> Dict[str, Any]:
    def _timeobj_to_seconds_local(t: Any) -> float:
        if pd.isna(t) or t is None:
            return float("nan")
        try:
            return float(int(t.hour) * 3600 + int(t.minute) * 60 + int(getattr(t, "second", 0)))
        except Exception:
            return float("nan")

    def _hhmm_to_seconds_local(hhmm: str) -> int:
        hhmm = (hhmm or "").strip()
        if not hhmm:
            return 0
        parts = hhmm.split(":")
        if len(parts) < 2:
            return 0
        try:
            hh = int(parts[0])
            mm = int(parts[1])
        except Exception:
            return 0
        return hh * 3600 + mm * 60

    def _build_terminal_usage(res_f: pd.DataFrame, top_n: int = 0) -> Dict[str, Any]:
        if res_f is None or res_f.empty:
            return {"labels": [], "manual": [], "carrete": [], "otros": [], "total": [], "top": {}}

        if "Terminal" not in res_f.columns or "Pulsos" not in res_f.columns:
            return {"labels": [], "manual": [], "carrete": [], "otros": [], "total": [], "top": {}}

        df = res_f.copy()

        df["Terminal"] = df["Terminal"].astype(str).str.strip()
        df["Terminal"] = df["Terminal"].replace({"nan": "", "None": ""})
        df = df[df["Terminal"] != ""].copy()
        if df.empty:
            return {"labels": [], "manual": [], "carrete": [], "otros": [], "total": [], "top": {}}

        df["Pulsos"] = pd.to_numeric(df["Pulsos"], errors="coerce").fillna(0).astype(int)

        if "Aplicación" in df.columns:
            df["Aplicación"] = df["Aplicación"].astype(str).str.strip()
        else:
            df["Aplicación"] = "Otros"

        g = df.groupby(["Terminal", "Aplicación"], dropna=False)["Pulsos"].sum()
        pivot = g.unstack(fill_value=0)

        cols = [str(c) for c in pivot.columns]
        manual_cols = [c for c in cols if "manual" in c.lower()]
        carrete_cols = [c for c in cols if "carrete" in c.lower()]

        manual = pivot[manual_cols].sum(axis=1) if manual_cols else (pivot.sum(axis=1) * 0)
        carrete = pivot[carrete_cols].sum(axis=1) if carrete_cols else (pivot.sum(axis=1) * 0)

        total_all = pivot.sum(axis=1)
        otros = (total_all - manual - carrete).clip(lower=0)

        out = pd.DataFrame(
            {
                "manual": manual.astype(int),
                "carrete": carrete.astype(int),
                "otros": otros.astype(int),
            }
        )
        out["total"] = (out["manual"] + out["carrete"] + out["otros"]).astype(int)
        out = out.sort_values("total", ascending=False)

        if top_n and top_n > 0:
            out = out.head(int(top_n))

        labels = out.index.astype(str).tolist()

        top = {}
        if len(out):
            first = out.iloc[0]
            top = {
                "terminal": str(out.index[0]),
                "manual": int(first["manual"]),
                "carrete": int(first["carrete"]),
                "otros": int(first["otros"]),
                "total": int(first["total"]),
            }

        return {
            "labels": labels,
            "manual": out["manual"].tolist(),
            "carrete": out["carrete"].tolist(),
            "otros": out["otros"].tolist(),
            "total": out["total"].tolist(),
            "top": top,
        }

    def _build_hourly_buckets(res_f: pd.DataFrame, par_f: pd.DataFrame) -> list[dict]:
        # ✅ IMPORTANTE: para que salgan horas con SOLO paradas (sin pulsos),
        # NO podemos devolver [] solo porque res_f venga vacío.
        if (res_f is None or res_f.empty) and (par_f is None or par_f.empty):
            return []

        if "Fecha" not in (res_f.columns if res_f is not None else []) and "Fecha" not in (
        par_f.columns if par_f is not None else []):
            return []

        # =========================
        # Pulsos por hora (si existe res_f)
        # =========================
        pulses_by_slot: Dict[tuple, int] = {}
        if res_f is not None and not res_f.empty and ("Fecha" in res_f.columns) and ("Hora" in res_f.columns):
            tmp = res_f.dropna(subset=["Fecha", "Hora"]).copy()
            tmp["Fecha_d"] = pd.to_datetime(tmp["Fecha"], errors="coerce").dt.date
            tmp["Hora_h"] = tmp["Hora"].apply(lambda t: int(t.hour) if hasattr(t, "hour") else np.nan)
            tmp["Hora_h"] = pd.to_numeric(tmp["Hora_h"], errors="coerce")
            tmp = tmp.dropna(subset=["Fecha_d", "Hora_h"]).copy()
            tmp["Hora_h"] = tmp["Hora_h"].astype(int)

            if not tmp.empty:
                tmp["Pulsos"] = pd.to_numeric(tmp.get("Pulsos", 0), errors="coerce").fillna(0).astype(int)
                pulses_by_slot = tmp.groupby(["Fecha_d", "Hora_h"])["Pulsos"].sum().to_dict()

        downtime_by_slot: Dict[tuple, float] = {}
        meal_by_slot: Dict[tuple, float] = {}

        # ✅ lo que usará el botón "Otro"
        other_secs_by_slot: Dict[tuple, Dict[str, float]] = {}
        other_desc_by_slot: Dict[tuple, Dict[str, str]] = {}

        def _strip_accents(s: str) -> str:
            return (
                (s or "")
                .replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
                .replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
                .replace("ñ", "n").replace("Ñ", "N")
            )

        def _norm_interval_key(s: str) -> str:
            s = _strip_accents(str(s or "").strip().lower())
            s = s.replace(" ", "_").replace("__", "_")
            return s.strip("_")

        def _interval_pair_key(interval_key: str) -> str:
            if interval_key == "finalizar":
                return "finalizar→inicio"
            if interval_key.endswith("_inicio"):
                base = interval_key[:-6]
                if base:
                    return f"{interval_key}→{base}_fin"
            return interval_key

        def _pretty_interval_desc(pair_key: str) -> str:
            if pair_key == "finalizar→inicio":
                return "Finalizar → Inicio"
            if "→" in pair_key:
                left, _right = pair_key.split("→", 1)
                base = left.replace("_inicio", "").replace("_", " ").strip()
                if base:
                    return f"{base.title()} (inicio → fin)"
            return pair_key

        # =========================
        # Partir paradas por hora
        # =========================
        if (
                par_f is not None
                and not par_f.empty
                and "Fecha" in par_f.columns
                and "Hora de Inicio" in par_f.columns
                and "downtime_s" in par_f.columns
        ):
            ptmp = par_f.dropna(subset=["Fecha", "Hora de Inicio"]).copy()
            if not ptmp.empty:
                ptmp["Fecha0"] = pd.to_datetime(ptmp["Fecha"], errors="coerce").dt.normalize()

                sec0 = ptmp["Hora de Inicio"].apply(_timeobj_to_seconds_local)
                sec0 = pd.to_numeric(sec0, errors="coerce").fillna(0).astype(int)

                ptmp["CausaCode"] = pd.to_numeric(ptmp.get("Causa", np.nan), errors="coerce")

                if "CausaDesc" in ptmp.columns:
                    ptmp["CausaDesc"] = ptmp["CausaDesc"].astype(str).str.strip()
                    ptmp.loc[ptmp["CausaDesc"].str.lower().isin(["nan", "none"]), "CausaDesc"] = ""
                else:
                    ptmp["CausaDesc"] = ""

                if "Tipo de Intervalo" in ptmp.columns:
                    ptmp["Tipo de Intervalo"] = ptmp["Tipo de Intervalo"].astype(str).str.strip()
                    ptmp.loc[ptmp["Tipo de Intervalo"].str.lower().isin(["nan", "none"]), "Tipo de Intervalo"] = ""
                else:
                    ptmp["Tipo de Intervalo"] = ""

                start_dt = ptmp["Fecha0"] + pd.to_timedelta(sec0, unit="s")
                dur_s = pd.to_numeric(ptmp["downtime_s"], errors="coerce").fillna(0.0).astype(float)
                end_dt = start_dt + pd.to_timedelta(dur_s, unit="s")

                for s, e, cc, desc, intervalo in zip(
                        start_dt.tolist(),
                        end_dt.tolist(),
                        ptmp["CausaCode"].tolist(),
                        ptmp["CausaDesc"].tolist(),
                        ptmp["Tipo de Intervalo"].tolist(),
                ):
                    if pd.isna(s) or pd.isna(e) or e <= s:
                        continue

                    code_int = None
                    try:
                        if not pd.isna(cc):
                            code_int = int(cc)
                    except Exception:
                        code_int = None

                    interval_key = _norm_interval_key(intervalo)
                    pair_key = _interval_pair_key(interval_key)

                    if code_int is not None:
                        key_code = str(code_int)
                        key_desc = (desc or "").strip()
                    else:
                        key_code = (pair_key or interval_key or "sin_codigo")
                        key_desc = _pretty_interval_desc(key_code)

                    is_meal = (code_int == 105)

                    cur = s
                    while cur < e:
                        hour_start = cur.replace(minute=0, second=0, microsecond=0)
                        next_hour = hour_start + pd.Timedelta(hours=1)
                        overlap_end = e if e < next_hour else next_hour

                        secs = float((overlap_end - cur).total_seconds())
                        slot = (hour_start.date(), int(hour_start.hour))

                        downtime_by_slot[slot] = downtime_by_slot.get(slot, 0.0) + secs
                        if is_meal:
                            meal_by_slot[slot] = meal_by_slot.get(slot, 0.0) + secs

                        # ✅ SOLO "Otro" (excluir 105)
                        if not is_meal:
                            if slot not in other_secs_by_slot:
                                other_secs_by_slot[slot] = {}
                                other_desc_by_slot[slot] = {}
                            other_secs_by_slot[slot][key_code] = other_secs_by_slot[slot].get(key_code, 0.0) + secs
                            if key_desc and (
                                    key_code not in other_desc_by_slot[slot] or not other_desc_by_slot[slot][key_code]
                            ):
                                other_desc_by_slot[slot][key_code] = key_desc

                        cur = overlap_end

        def _to_items(secs_map: Dict[str, float], desc_map: Dict[str, str], top_n: int = 10) -> List[Dict[str, Any]]:
            if not secs_map:
                return []
            items_sorted = sorted(secs_map.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[: int(top_n)]

            out: List[Dict[str, Any]] = []
            for code, sec in items_sorted:
                sec_f = float(sec or 0.0)
                out.append(
                    {
                        "code": str(code),
                        "desc": str(desc_map.get(code, "")),
                        "seconds": sec_f,
                        "hhmm": _fmt_hhmm(sec_f),
                    }
                )
            return out

        # ✅ incluir horas con pulsos o con paradas (o ambas)
        slot_keys = set(pulses_by_slot.keys())
        slot_keys |= set(downtime_by_slot.keys())
        slot_keys |= set(meal_by_slot.keys())

        active_slots = sorted(slot_keys, key=lambda x: (x[0], x[1]))

        buckets: list[dict] = []
        for (d, h) in active_slots:
            pulses = int(pulses_by_slot.get((d, h), 0))

            dead = float(downtime_by_slot.get((d, h), 0.0))
            meal = float(meal_by_slot.get((d, h), 0.0))

            dead = min(max(dead, 0.0), 3600.0)
            meal = min(max(meal, 0.0), dead)

            prod = max(0.0, 3600.0 - dead)
            other_dead = max(0.0, dead - meal)

            slot = (d, h)
            other_items = _to_items(other_secs_by_slot.get(slot, {}), other_desc_by_slot.get(slot, {}), top_n=10)

            buckets.append(
                {
                    "hourLabel": f"{int(h):02d}:00",
                    "cut": pulses,
                    "meters": 0.0,
                    "prodSec": prod,
                    "deadSec": dead,
                    "mealSec": meal,
                    "otherDeadSec": other_dead,
                    "len_mix": None,
                    "other_causes": other_items,
                }
            )

        return buckets

    def _fill_missing_hourly_buckets_gaps_only(
            buckets: list[dict],
            step_hours: int = 1,
    ) -> list[dict]:
        """
        Rellena SOLO los huecos entre la primera y la última hora con registro.
        No agrega antes del primer registro ni después del último.
        """
        if not buckets:
            return []

        def _h_from_label(lbl: str) -> Optional[int]:
            try:
                # "07:00" -> 7
                return int(str(lbl).strip().split(":")[0])
            except Exception:
                return None

        # mapa por hora
        bmap: Dict[int, dict] = {}
        hours: List[int] = []
        for b in buckets:
            h = _h_from_label(b.get("hourLabel", ""))
            if h is None:
                continue
            bmap[h] = b
            hours.append(h)

        if not hours:
            return buckets

        h_min = min(hours)
        h_max = max(hours)

        out: list[dict] = []
        for h in range(h_min, h_max + 1, int(step_hours)):
            if h in bmap:
                bb = dict(bmap[h])
                bb["_missing"] = False
                out.append(bb)
            else:
                out.append(
                    {
                        "hourLabel": f"{h:02d}:00",
                        "cut": 0,
                        "meters": 0.0,
                        "prodSec": 0.0,
                        "deadSec": 0.0,
                        "mealSec": 0.0,
                        "otherDeadSec": 0.0,
                        "len_mix": None,
                        "other_causes": [],
                        "_missing": True,  # ✅ para pintar gris
                    }
                )
        return out

    # =========================
    # Cargar datos
    # =========================
    res_df = load_resumen_pulsos(path, machine=machine, machine_id=machine_id)
    par_df, _ = load_paradas(path, machine=machine, machine_id=machine_id)

    res_f = _filter_period(res_df, period, period_value, "Fecha")
    par_f = _filter_period(par_df, period, period_value, "Fecha")

    # filtro por operaria
    operator = (operator or "").strip()
    if operator:
        if "Nombre" in res_f.columns:
            res_f = res_f[res_f["Nombre"].astype(str).str.strip() == operator]
        if "Nombre" in par_f.columns:
            par_f = par_f[par_f["Nombre"].astype(str).str.strip() == operator]

    # filtro por hora (Resumen: Hora)
    time_start = (time_start or "").strip()
    time_end = (time_end or "").strip()
    if time_start and time_end and "Hora" in res_f.columns:
        s0 = _hhmm_to_seconds_local(time_start)
        s1 = _hhmm_to_seconds_local(time_end)
        if s1 < s0:
            s0, s1 = s1, s0
        sec_series = res_f["Hora"].apply(_timeobj_to_seconds_local)
        res_f = res_f[(sec_series >= s0) & (sec_series <= s1)]

    # filtro por hora (Paradas: Hora de Inicio)
    if time_start and time_end and "Hora de Inicio" in par_f.columns:
        s0 = _hhmm_to_seconds_local(time_start)
        s1 = _hhmm_to_seconds_local(time_end)
        if s1 < s0:
            s0, s1 = s1, s0
        sec_ini = par_f["Hora de Inicio"].apply(_timeobj_to_seconds_local)
        par_f = par_f[(sec_ini >= s0) & (sec_ini <= s1)]

    # quitar marcadores (si existe)
    if "Tipo de Intervalo" in par_f.columns:
        par_f = par_f[~par_f["Tipo de Intervalo"].astype(str).str.contains("marcador", case=False, na=False)].copy()
    else:
        par_f = par_f.copy()

    # downtime_s
    # downtime_s (robusto)
    dur_raw = par_f.get("Duracion", pd.Series([None] * len(par_f), index=par_f.index, dtype="object"))

    # intenta parse directo "HH:MM:SS" / "MM:SS" / timedelta
    dt = pd.to_timedelta(dur_raw, errors="coerce")
    secs = dt.dt.total_seconds()

    # ✅ fallback: si Duracion no existe o no parsea, intenta con "Hora de Fin" - "Hora de Inicio"
    # ✅ fallback: si Duracion no existe o no parsea, intenta con "Hora de Fin" - "Hora de Inicio"
    if secs.isna().all() and ("Hora de Inicio" in par_f.columns) and ("Hora de Fin" in par_f.columns):
        ini = par_f["Hora de Inicio"].apply(_timeobj_to_seconds_local)
        fin = par_f["Hora de Fin"].apply(_timeobj_to_seconds_local)

        ini = pd.to_numeric(ini, errors="coerce")
        fin = pd.to_numeric(fin, errors="coerce")

        diff = fin - ini
        diff = np.where(diff < 0, diff + 86400, diff)  # cruza medianoche
        secs = pd.Series(diff, index=par_f.index, dtype="float64")

    par_f["downtime_s"] = pd.to_numeric(secs, errors="coerce").fillna(0.0).astype(float)

    # -------------------------
    # Pulsos
    # -------------------------
    pulses_total = int(res_f["Pulsos"].sum()) if "Pulsos" in res_f.columns else 0

    pulses_by_type: Dict[str, int] = {}
    if "Aplicación" in res_f.columns and "Pulsos" in res_f.columns and not res_f.empty:
        pulses_by_type = res_f.groupby("Aplicación")["Pulsos"].sum().astype(int).to_dict()

    carrete = 0
    manual = 0
    for k, v in (pulses_by_type or {}).items():
        kk = str(k).strip().lower()
        if "carrete" in kk:
            carrete += int(v)
        elif "manual" in kk:
            manual += int(v)

    total_for_pct = float(carrete + manual) if (carrete + manual) > 0 else float(pulses_total)
    pct_carrete = (carrete / total_for_pct) if total_for_pct > 0 else 0.0
    pct_manual = (manual / total_for_pct) if total_for_pct > 0 else 0.0

    def _hhmmss_to_seconds(s: str) -> int:
        s = (s or "").strip()
        if not s:
            return 0
        parts = s.split(":")
        try:
            if len(parts) == 2:
                hh, mm = int(parts[0]), int(parts[1])
                return hh * 3600 + mm * 60
            if len(parts) >= 3:
                hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
                return hh * 3600 + mm * 60 + ss
        except Exception:
            return 0
        return 0

    BASE_MON_THU = _hhmmss_to_seconds("09:08:00")  # 9h08
    BASE_FRI_SAT = _hhmmss_to_seconds("06:53:00")  # 6h53

    def _expected_day_seconds(day_date, day_slot_hours: int) -> int:
        """
        Regla:
        - Lun-Jue: base 9:08 si hay >=10 horas registradas; si hay 11 -> +1h, etc.
          Si hay <10, ignorar y dejar 9:08.
        - Vie/Sab: base 6:53 si hay >=10 horas registradas; si hay 11 -> +1h, etc.
          Si hay <10, ignorar y dejar 6:53.
        - Dom: por defecto 0 (ajústalo si aplica).
        """
        if day_date is None:
            return 0
        wd = int(pd.Timestamp(day_date).weekday())  # 0=Lun ... 6=Dom

        if wd in (0, 1, 2, 3):  # Lun-Jue
            base = BASE_MON_THU
        elif wd in (4, 5):  # Vie-Sab
            base = BASE_FRI_SAT
        else:  # Dom
            return 0

        # condición de "debe haber registro de 10 horas min"
        if day_slot_hours < 10:
            return base

        extra_h = max(0, day_slot_hours - 10)  # 11->+1, 12->+2, ...
        return base + extra_h * 3600

    # -------------------------
    # Horas activas (slots)
    # -------------------------
    # -------------------------
    # Horas activas (slots)  ✅ incluye res_f (pulsos) + par_f (paradas)
    # -------------------------
    active_slots: set[tuple] = set()

    # slots desde pulsos
    if "Fecha" in res_f.columns and "Hora" in res_f.columns and not res_f.empty:
        tmp = res_f.dropna(subset=["Fecha", "Hora"]).copy()
        tmp["Fecha_d"] = pd.to_datetime(tmp["Fecha"], errors="coerce").dt.date
        tmp["Hora_h"] = tmp["Hora"].apply(lambda t: int(t.hour) if hasattr(t, "hour") else np.nan)
        tmp = tmp.dropna(subset=["Fecha_d", "Hora_h"])
        active_slots |= set(zip(tmp["Fecha_d"].tolist(), tmp["Hora_h"].astype(int).tolist()))

    # slots desde paradas (Hora de Inicio)
    if "Fecha" in par_f.columns and "Hora de Inicio" in par_f.columns and not par_f.empty:
        ptmp = par_f.dropna(subset=["Fecha", "Hora de Inicio"]).copy()
        ptmp["Fecha_d"] = pd.to_datetime(ptmp["Fecha"], errors="coerce").dt.date
        ptmp["Hora_h"] = ptmp["Hora de Inicio"].apply(lambda t: int(t.hour) if hasattr(t, "hour") else np.nan)
        ptmp = ptmp.dropna(subset=["Fecha_d", "Hora_h"])
        active_slots |= set(zip(ptmp["Fecha_d"].tolist(), ptmp["Hora_h"].astype(int).tolist()))

    slot_hours_total = int(len(active_slots))

    # -------------------------
    # Horas activas esperadas (según regla) + No registrado
    # -------------------------
    slots_by_day: Dict[Any, int] = {}
    for (d, _h) in active_slots:
        slots_by_day[d] = slots_by_day.get(d, 0) + 1

    expected_active_s = 0
    for d, day_slots in slots_by_day.items():
        expected_active_s += _expected_day_seconds(d, int(day_slots))

    # Horas activas (lo que pides en el KPI) = esperado según regla
    active_available_s = float(expected_active_s)

    # =========================================================
    # ✅ NUEVO: Horas activas planificadas (segundos)
    # Lun–Jue: 09:08, requiere >=10 horas registradas para validar extra
    # Vie–Sáb: 06:53, requiere >=7 horas registradas (ajustable)
    # Si supera el mínimo, suma +1h por cada hora extra
    # =========================================================
    def _planned_active_seconds_from_slots(res_filtered: pd.DataFrame) -> float:
        if res_filtered is None or res_filtered.empty:
            return 0.0
        if "Fecha" not in res_filtered.columns or "Hora" not in res_filtered.columns:
            return 0.0

        tmp2 = res_filtered.dropna(subset=["Fecha", "Hora"]).copy()
        tmp2["Fecha_d"] = pd.to_datetime(tmp2["Fecha"], errors="coerce").dt.date
        tmp2["Hora_h"] = tmp2["Hora"].apply(lambda t: int(t.hour) if hasattr(t, "hour") else np.nan)
        tmp2["Hora_h"] = pd.to_numeric(tmp2["Hora_h"], errors="coerce")
        tmp2 = tmp2.dropna(subset=["Fecha_d", "Hora_h"]).copy()
        if tmp2.empty:
            return 0.0
        tmp2["Hora_h"] = tmp2["Hora_h"].astype(int)

        # horas registradas (slots únicos) por día
        g = tmp2.groupby("Fecha_d")["Hora_h"].nunique(dropna=True)

        mon_thu_base = 9*3600 + 8*60    # 09:08
        fri_sat_base = 6*3600 + 53*60   # 06:53

        mon_thu_min_hours = 10
        fri_sat_min_hours = 7  # <-- AJUSTA si tu mínimo real es otro

        total = 0.0
        for d, n in g.items():
            wd = pd.Timestamp(d).weekday()  # Mon=0 ... Sun=6

            if wd in (4, 5):  # Vie/Sáb
                base = fri_sat_base
                min_h = fri_sat_min_hours
            else:             # Lun–Jue (y domingo si cae)
                base = mon_thu_base
                min_h = mon_thu_min_hours

            extra_h = max(0, int(n) - int(min_h))
            total += float(base + extra_h*3600)

        return float(total)

    # ✅ Usa la regla esperada por día (y NO _planned_active_seconds_from_slots)
    # expected_active_s ya lo calculaste arriba con _expected_day_seconds(...)
    active_available_s = float(expected_active_s)

    # -------------------------
    # Tiempos (Plan/No plan)
    # -------------------------
    par_f["CausaCode"] = pd.to_numeric(par_f.get("Causa", np.nan), errors="coerce")
    planned_s = float(par_f.loc[(par_f["CausaCode"] >= 100) & (par_f["CausaCode"] <= 199), "downtime_s"].sum()) \
        if "downtime_s" in par_f.columns else 0.0
    unplanned_s = float(par_f.loc[(par_f["CausaCode"] >= 200) & (par_f["CausaCode"] <= 299), "downtime_s"].sum()) \
        if "downtime_s" in par_f.columns else 0.0

    # -------------------------
    # 105 (Comida)
    # -------------------------
    meal_total_s = float(par_f.loc[(par_f["CausaCode"] == 105), "downtime_s"].sum()) \
        if "downtime_s" in par_f.columns else 0.0

    # -------------------------
    # Horas activas esperadas + No registrado
    # -------------------------
    # tiempo "registrado" por tarjetas = cantidad de slots activos * 3600
    registered_slot_s = float(slot_hours_total) * 3600.0

    # downtime total real del periodo
    dead_total_s = float(par_f["downtime_s"].sum()) if "downtime_s" in par_f.columns else 0.0

    # ✅ Tiempo efectivo (base) = registrado - paradas
    productive_s = max(0.0, registered_slot_s - dead_total_s)

    # ✅ “Otro” sin 105 (muerto excluyendo comida)  ✅ (DEFINIR ANTES DE USAR)
    other_total_s = max(0.0, dead_total_s - meal_total_s)

    # ✅ Regla nueva: No registrado = Disponible - (Efectivo + Otro sin 105)
    no_reg_s = max(0.0, float(active_available_s) - (float(productive_s) + float(other_total_s)))

    # ✅ % de “Otro” sobre el total complementado (Efectivo + Otro + 105 + No registrado)
    total_available_s = float(productive_s) + float(other_total_s) + float(meal_total_s) + float(no_reg_s)
    pct_other = (other_total_s / total_available_s * 100.0) if total_available_s > 0 else 0.0

    # -------------------------
    # Pareto
    # -------------------------
    pareto = build_pareto_paradas(par_f, top_n=10)

    # ✅ Para click en "Otro" (tarjetas grises)
    other_causes = pareto.get("items", []) if isinstance(pareto, dict) else []
    if other_causes is None:
        other_causes = []

    # -------------------------
    # Terminales más usadas
    # -------------------------
    terminal_usage = _build_terminal_usage(res_f, top_n=0)

    # -------------------------
    # Productividad por hora (SOLO DAY)
    # -------------------------
    hourly_buckets: list[dict] = []
    if (period or "").strip().lower() == "day":
        hourly_buckets = _build_hourly_buckets(res_f, par_f)
        # ✅ solo rellena huecos entre registros (una sola vez)
        hourly_buckets = _fill_missing_hourly_buckets_gaps_only(hourly_buckets)

    # =========================================================
    # ✅ REGLA: KPIs de tiempo = suma exacta de lo que muestran las tarjetas (hourly_buckets)
    # =========================================================
    if hourly_buckets:
        sum_prod_s = float(sum(float(b.get("prodSec", 0.0) or 0.0) for b in hourly_buckets))
        sum_dead_s = float(sum(float(b.get("deadSec", 0.0) or 0.0) for b in hourly_buckets))
        sum_meal_s = float(sum(float(b.get("mealSec", 0.0) or 0.0) for b in hourly_buckets))

        # ✅ Tiempo efectivo = suma de "Efectivo" mostrado en tarjetas
        productive_s = sum_prod_s

        # ✅ downtime total = suma de "Muerto" mostrado en tarjetas
        dead_total_s = sum_dead_s

        # ✅ comida total = suma de "Comida" mostrado en tarjetas
        meal_total_s = sum_meal_s

        # ✅ “Otro” sin 105 (muerto excluyendo comida) coherente con tarjetas
        other_total_s = max(0.0, sum_dead_s - sum_meal_s)

        # ✅ Regla nueva: No registrado = Disponible - (Efectivo + Otro sin 105)
        no_reg_s = max(0.0, float(active_available_s) - (float(productive_s) + float(other_total_s)))

        # ✅ Recalcular % de Otro sobre el total complementado
        total_available_s = float(productive_s) + float(other_total_s) + float(meal_total_s) + float(no_reg_s)
        pct_other = (other_total_s / total_available_s * 100.0) if total_available_s > 0 else 0.0

    # -------------------------
    # KPIs UI
    # -------------------------
    kpis_ui: Dict[str, str] = {
        "Crimpados": _fmt_int_es(pulses_total),
        "Crimpados Carrete": f"{_fmt_int_es(carrete)} ({_fmt_pct_es(pct_carrete)})",
        "Crimpados Manual": f"{_fmt_int_es(manual)} ({_fmt_pct_es(pct_manual)})",

        # ✅ tu frontend usa "Horas activas" (no "Horas Disponibles")
        "Horas activas": _fmt_hhmm(active_available_s),

        # ✅ “Otro” EXCLUYE 105, y muestra 105 + No registrado + % en el panel derecho
        "Otro Tiempo de Ciclo (Muerto)": (
            f"{_fmt_hhmm(other_total_s)}"
            f"||105: {_fmt_hhmm(meal_total_s)} | No registrado: {_fmt_hhmm(no_reg_s)} | {pct_other:.1f}%"
        ),

        "Tiempo Efectivo": _fmt_hhmm(productive_s),
        "Paradas planeadas (HH:MM)": _fmt_hhmm(planned_s),
        "Paradas no planeadas (HH:MM)": _fmt_hhmm(unplanned_s),
    }

    return {
        "machine": machine,
        "subcat": subcat or "",
        "operator": operator,
        "time_start": time_start,
        "time_end": time_end,
        "period": period,
        "period_value": period_value,
        "rows_total": int(len(res_f) + len(par_f)),
        "kpis_ui": kpis_ui,
        "pareto": pareto,

        "other_causes": other_causes,  # compat

        "kpis": {
            "hourly_buckets": hourly_buckets,
            "other_causes": other_causes,
        },
        "terminal_usage": terminal_usage,
        "machine_id": _norm_machine_id(machine_id),
    }


def analyze_aplicacion1(
    path: str,
    period: str,
    period_value: str,
    machine: str = "APLICACION",
    subcat: str = "",
    machine_id: str = "",
):
    """
    Alias de compatibilidad:
    Retorna (result, err) para encajar con el patrón existente.
    """
    try:
        result = run_aplicacion_analysis(
            path=path,
            period=period,
            period_value=period_value,
            machine=machine,
            subcat=subcat,
            machine_id=machine_id,
        )
        return result, None
    except Exception as e:
        return None, str(e)

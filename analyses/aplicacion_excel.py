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
# OEE esperado crimpado
# ============================================================
# Base para Aplicación M1/M2 y Unión M2.
OEE_ESPERADO_CRIMPADO_BASE = 0.90

# Cambia SOLO este valor para Unión - Máquina 1.
# Ejemplos: 0.95 = 95%, 0.90 = 90%, 0.85 = 85%.
OEE_ESPERADO_UNION1 = 0.95


def _crimpado_oee_esperado(machine: str = "", machine_id: str = "") -> float:
    """Devuelve el OEE esperado. Solo Unión M1 tiene valor independiente."""
    m = str(machine or "").strip().upper()
    mid = str(machine_id or "").strip()
    if m == "UNION" and mid == "1":
        return float(OEE_ESPERADO_UNION1)
    return float(OEE_ESPERADO_CRIMPADO_BASE)


# ============================================================
# Factor adicional de meta crimpado
# ============================================================
# Cambia SOLO este valor si quieres subir/bajar la meta de Unión - Máquina 1.
# Ejemplos: 1.15 = sube 15%, 1.10 = sube 10%, 1.00 = sin ajuste.
FACTOR_META_UNION1 = 1.25


def _crimpado_meta_factor(machine: str = "", machine_id: str = "") -> float:
    """Devuelve el factor de meta. Solo Unión M1 tiene ajuste independiente."""
    m = str(machine or "").strip().upper()
    mid = str(machine_id or "").strip()
    if m == "UNION" and mid == "1":
        return float(FACTOR_META_UNION1)
    return 1.0


def _crimpado_meta_plan_time_h(tp_h: float, tw_h: float, machine: str = "", machine_id: str = "") -> float:
    """
    Tiempo base para calcular la producción planeada.

    Regla solicitada:
    - Unión M1 usa TW, para que las paradas TX reduzcan la meta de la hora.
    - Las demás máquinas conservan TP, como venían trabajando.
    """
    if _crimpado_is_union1(machine, machine_id):
        return max(0.0, float(tw_h or 0.0))
    return max(0.0, float(tp_h or 0.0))


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

    df["CausaCode"] = pd.to_numeric(df["CausaCode"], errors="coerce")
    df = df[df["CausaCode"] != 105].copy()

    if df.empty:
        return {"total_s": 0.0, "items": []}

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
        if (res_f is None or res_f.empty) and (par_f is None or par_f.empty):
            return []

        if "Fecha" not in (res_f.columns if res_f is not None else []) and "Fecha" not in (
            par_f.columns if par_f is not None else []
        ):
            return []

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
            other_dead = max(0.0, dead - meal)

            # Unión 1: garantizar desayuno/almuerzo no pagos aunque no estén
            # registrados explícitamente como parada 105.
            dead, meal, other_dead, prod = _crimpado_apply_fixed_non_paid_to_parts(
                dead, meal, other_dead, h, machine=machine, machine_id=machine_id
            )

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
        if not buckets:
            return []

        def _h_from_label(lbl: str) -> Optional[int]:
            try:
                return int(str(lbl).strip().split(":")[0])
            except Exception:
                return None

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
                        "_missing": True,
                    }
                )
        return out

    def _push_last_sec(day_map: Dict[Any, float], day_value: Any, sec_value: Any) -> None:
        if pd.isna(day_value) or pd.isna(sec_value):
            return
        try:
            sec_f = float(sec_value)
        except Exception:
            return
        if pd.isna(sec_f):
            return
        prev = day_map.get(day_value)
        if prev is None or sec_f > float(prev):
            day_map[day_value] = sec_f

    def _extra_hours_from_cutoff_strict(last_sec: Optional[float], cutoff_sec: int) -> int:
        """
        Suma +1 hora solo si PASA del corte.
        Ej:
        - corte 14:30
        - 14:30 -> 0
        - 14:31 -> 1
        - 15:30 -> 1
        - 15:31 -> 2
        """
        if last_sec is None or pd.isna(last_sec):
            return 0
        try:
            sec_f = float(last_sec)
        except Exception:
            return 0
        if sec_f <= float(cutoff_sec):
            return 0
        return int(((sec_f - float(cutoff_sec)) + 3599) // 3600)

    # =========================
    # Cargar datos
    # =========================
    res_df = load_resumen_pulsos(path, machine=machine, machine_id=machine_id)
    par_df, _ = load_paradas(path, machine=machine, machine_id=machine_id)

    res_f = _filter_period(res_df, period, period_value, "Fecha")
    par_f = _filter_period(par_df, period, period_value, "Fecha")

    operator = (operator or "").strip()
    if operator:
        if "Nombre" in res_f.columns:
            res_f = res_f[res_f["Nombre"].astype(str).str.strip() == operator]
        if "Nombre" in par_f.columns:
            par_f = par_f[par_f["Nombre"].astype(str).str.strip() == operator]

    time_start = (time_start or "").strip()
    time_end = (time_end or "").strip()
    if time_start and time_end and "Hora" in res_f.columns:
        s0 = _hhmm_to_seconds_local(time_start)
        s1 = _hhmm_to_seconds_local(time_end)
        if s1 < s0:
            s0, s1 = s1, s0
        sec_series = res_f["Hora"].apply(_timeobj_to_seconds_local)
        res_f = res_f[(sec_series >= s0) & (sec_series <= s1)]

    if time_start and time_end and "Hora de Inicio" in par_f.columns:
        s0 = _hhmm_to_seconds_local(time_start)
        s1 = _hhmm_to_seconds_local(time_end)
        if s1 < s0:
            s0, s1 = s1, s0
        sec_ini = par_f["Hora de Inicio"].apply(_timeobj_to_seconds_local)
        par_f = par_f[(sec_ini >= s0) & (sec_ini <= s1)]

    if "Tipo de Intervalo" in par_f.columns:
        par_f = par_f[~par_f["Tipo de Intervalo"].astype(str).str.contains("marcador", case=False, na=False)].copy()
    else:
        par_f = par_f.copy()

    dur_raw = par_f.get("Duracion", pd.Series([None] * len(par_f), index=par_f.index, dtype="object"))
    dt = pd.to_timedelta(dur_raw, errors="coerce")
    secs = dt.dt.total_seconds()

    if secs.isna().all() and ("Hora de Inicio" in par_f.columns) and ("Hora de Fin" in par_f.columns):
        ini = par_f["Hora de Inicio"].apply(_timeobj_to_seconds_local)
        fin = par_f["Hora de Fin"].apply(_timeobj_to_seconds_local)

        ini = pd.to_numeric(ini, errors="coerce")
        fin = pd.to_numeric(fin, errors="coerce")

        diff = fin - ini
        diff = np.where(diff < 0, diff + 86400, diff)
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

    # =========================================================
    # TIEMPO PAGADO / HORAS ACTIVAS
    # L-J: 9.25 h fijas; solo si pasa de las 5:00 pm suma horas
    # V:   7.00 h fijas; solo si pasa de las 2:30 pm suma horas
    # S:   5.00 h fijas; solo si pasa de las 12:30 pm suma horas
    # =========================================================
    BASE_LJ_SEC = int(round(9.25 * 3600))       # 09:15
    BASE_FRI_SEC = 7 * 3600                     # 07:00
    BASE_SAT_SEC = 5 * 3600                     # 05:00

    CUTOFF_LJ_SEC = 17 * 3600                   # 5:00 pm
    CUTOFF_FRI_SEC = (14 * 3600) + (30 * 60)    # 2:30 pm
    CUTOFF_SAT_SEC = (12 * 3600) + (30 * 60)    # 12:30 pm

    def _expected_day_seconds(day_date: Any, last_sec: Optional[float] = None) -> int:
        if day_date is None:
            return 0

        wd = int(pd.Timestamp(day_date).weekday())  # 0=lun ... 6=dom

        if wd in (0, 1, 2, 3):  # Lun-Jue
            extra_h = _extra_hours_from_cutoff_strict(last_sec, CUTOFF_LJ_SEC)
            return int(BASE_LJ_SEC + extra_h * 3600)

        elif wd == 4:  # Viernes
            extra_h = _extra_hours_from_cutoff_strict(last_sec, CUTOFF_FRI_SEC)
            return int(BASE_FRI_SEC + extra_h * 3600)

        elif wd == 5:  # Sábado
            extra_h = _extra_hours_from_cutoff_strict(last_sec, CUTOFF_SAT_SEC)
            return int(BASE_SAT_SEC + extra_h * 3600)

        return 0

    # -------------------------
    # Horas activas (slots) + última hora registrada por día
    # -------------------------
    active_slots = set()
    last_sec_by_day: Dict[Any, float] = {}

    if "Fecha" in res_f.columns and "Hora" in res_f.columns and not res_f.empty:
        tmp = res_f.dropna(subset=["Fecha", "Hora"]).copy()
        tmp["Fecha_d"] = pd.to_datetime(tmp["Fecha"], errors="coerce").dt.date
        tmp["Hora_h"] = tmp["Hora"].apply(lambda t: int(t.hour) if hasattr(t, "hour") else np.nan)
        tmp["Hora_s"] = tmp["Hora"].apply(_timeobj_to_seconds_local)
        tmp = tmp.dropna(subset=["Fecha_d", "Hora_h"]).copy()

        active_slots |= set(
            zip(
                tmp["Fecha_d"].tolist(),
                pd.to_numeric(tmp["Hora_h"], errors="coerce").fillna(-1).astype(int).tolist()
            )
        )

        for d, sec in zip(tmp["Fecha_d"].tolist(), tmp["Hora_s"].tolist()):
            _push_last_sec(last_sec_by_day, d, sec)

    if "Fecha" in par_f.columns and not par_f.empty:
        ptmp = par_f.copy()
        ptmp["Fecha_d"] = pd.to_datetime(ptmp["Fecha"], errors="coerce").dt.date

        if "Hora de Inicio" in ptmp.columns:
            ptmp["HoraIni_h"] = ptmp["Hora de Inicio"].apply(lambda t: int(t.hour) if hasattr(t, "hour") else np.nan)
            ptmp["HoraIni_s"] = ptmp["Hora de Inicio"].apply(_timeobj_to_seconds_local)
            ptmp2 = ptmp.dropna(subset=["Fecha_d", "HoraIni_h"]).copy()

            active_slots |= set(
                zip(
                    ptmp2["Fecha_d"].tolist(),
                    pd.to_numeric(ptmp2["HoraIni_h"], errors="coerce").fillna(-1).astype(int).tolist()
                )
            )

            for d, sec in zip(ptmp["Fecha_d"].tolist(), ptmp["HoraIni_s"].tolist()):
                _push_last_sec(last_sec_by_day, d, sec)

        if "Hora de Fin" in ptmp.columns:
            ptmp["HoraFin_s"] = ptmp["Hora de Fin"].apply(_timeobj_to_seconds_local)
            for d, sec in zip(ptmp["Fecha_d"].tolist(), ptmp["HoraFin_s"].tolist()):
                _push_last_sec(last_sec_by_day, d, sec)

    slot_hours_total = int(len(active_slots))

    active_days = sorted({d for (d, _h) in active_slots if d is not None})

    expected_active_s = 0
    for d in active_days:
        expected_active_s += _expected_day_seconds(d, last_sec_by_day.get(d))

    active_available_s = float(expected_active_s)

    # -------------------------
    # Tiempos (Plan/No plan)
    # -------------------------
    par_f["CausaCode"] = pd.to_numeric(par_f.get("Causa", np.nan), errors="coerce")

    planned_s = float(
        par_f.loc[(par_f["CausaCode"] >= 100) & (par_f["CausaCode"] <= 199), "downtime_s"].sum()
    ) if "downtime_s" in par_f.columns else 0.0

    unplanned_s = float(
        par_f.loc[(par_f["CausaCode"] >= 200) & (par_f["CausaCode"] <= 299), "downtime_s"].sum()
    ) if "downtime_s" in par_f.columns else 0.0

    meal_total_s = float(
        par_f.loc[(par_f["CausaCode"] == 105), "downtime_s"].sum()
    ) if "downtime_s" in par_f.columns else 0.0

    parada104_s = float(
        par_f.loc[(par_f["CausaCode"] == 104), "downtime_s"].sum()
    ) if "downtime_s" in par_f.columns else 0.0

    registered_slot_s = float(slot_hours_total) * 3600.0
    dead_total_s = float(par_f["downtime_s"].sum()) if "downtime_s" in par_f.columns else 0.0

    productive_s = max(0.0, registered_slot_s - dead_total_s)
    other_total_s = max(0.0, dead_total_s - meal_total_s)

    no_reg_s = max(0.0, float(active_available_s) - (float(productive_s) + float(other_total_s)))

    total_available_s = float(productive_s) + float(other_total_s) + float(meal_total_s) + float(no_reg_s)
    pct_other = (other_total_s / total_available_s * 100.0) if total_available_s > 0 else 0.0

    # -------------------------
    # Pareto (excluye 105)
    # -------------------------
    par_f_pareto = par_f.copy()
    if "CausaCode" not in par_f_pareto.columns:
        par_f_pareto["CausaCode"] = pd.to_numeric(par_f_pareto.get("Causa", np.nan), errors="coerce")
    par_f_pareto = par_f_pareto[par_f_pareto["CausaCode"] != 105].copy()

    pareto = build_pareto_paradas(par_f_pareto, top_n=10)

    other_causes = pareto.get("items", []) if isinstance(pareto, dict) else []
    if other_causes is None:
        other_causes = []

    terminal_usage = _build_terminal_usage(res_f, top_n=0)

    hourly_buckets: list[dict] = []
    if (period or "").strip().lower() == "day":
        hourly_buckets = _build_hourly_buckets(res_f, par_f)
        hourly_buckets = _fill_missing_hourly_buckets_gaps_only(hourly_buckets)

    if hourly_buckets:
        sum_prod_s = float(sum(float(b.get("prodSec", 0.0) or 0.0) for b in hourly_buckets))
        sum_dead_s = float(sum(float(b.get("deadSec", 0.0) or 0.0) for b in hourly_buckets))
        sum_meal_s = float(sum(float(b.get("mealSec", 0.0) or 0.0) for b in hourly_buckets))

        productive_s = sum_prod_s
        dead_total_s = sum_dead_s
        meal_total_s = sum_meal_s
        other_total_s = max(0.0, sum_dead_s - sum_meal_s)

        no_reg_s = max(0.0, float(active_available_s) - (float(productive_s) + float(other_total_s)))

        total_available_s = float(productive_s) + float(other_total_s) + float(meal_total_s) + float(no_reg_s)
        pct_other = (other_total_s / total_available_s * 100.0) if total_available_s > 0 else 0.0

    pct_eff = (productive_s / active_available_s * 100.0) if active_available_s > 0 else 0.0

    # =========================================================
    # ✅ MODELO OEE CRIMPADO PARA DASHBOARD
    # =========================================================
    # El Excel exportado calcula PP, PB, TP, TX, TW, TC, VN, TCN, TCE,
    # ID, IEO, IC, OEE y PA. Aquí se replica el mismo modelo para que
    # el dashboard muestre los mismos indicadores sin esperar la descarga.
    OEE_ESPERADO = _crimpado_oee_esperado(machine, machine_id)

    def _safe_div_num(a: float, b: float, default: float = 0.0) -> float:
        try:
            a = float(a)
            b = float(b)
            if b == 0 or pd.isna(b):
                return default
            return float(a / b)
        except Exception:
            return default

    def _fmt_dec_es(x: Any, nd: int = 2) -> str:
        try:
            xf = float(x)
            if pd.isna(xf):
                xf = 0.0
        except Exception:
            xf = 0.0
        return f"{xf:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def _fmt_pct_from_ratio(x: float) -> str:
        return f"{(float(x or 0.0) * 100.0):.1f}%"

    def _fmt_units_per_hour(x: Any) -> str:
        try:
            xf = float(x)
            if pd.isna(xf) or xf <= 0:
                return "0 unid/h"
            return f"{int(round(xf)):,}".replace(",", ".") + " unid/h"
        except Exception:
            return "0 unid/h"

    # Tabla RESUMEN_TIEMPOS => TC nominal por terminal en horas decimales.
    try:
        resumen_tiempos_df = load_resumen_tiempos(path, machine=machine, machine_id=machine_id)
        tc_lookup, tc_default_h_uni = _crimpado_build_tc_lookup(resumen_tiempos_df)
    except Exception:
        tc_lookup, tc_default_h_uni = {}, None

    # Unión 1 no siempre tiene RESUMEN_TIEMPOS. En ese caso se calcula un
    # tiempo estándar nominal histórico único para la terminal de esa máquina:
    # TC_nom_UNION1 = SUM(TW_h) / SUM(PN_h).
    tc_nominal_history_h_uni = _crimpado_estimate_tc_nominal_from_history(
        res_df=res_df,
        par_df=par_df,
        machine=machine,
        machine_id=machine_id,
        operator=operator,
    )
    if (tc_default_h_uni is None or float(tc_default_h_uni or 0.0) <= 0.0) and tc_nominal_history_h_uni:
        tc_default_h_uni = float(tc_nominal_history_h_uni)

    def _crimpado_dashboard_rows_for_day(day_value: Any, buckets_day: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Arma filas equivalentes al Excel, pero usando el res_f/par_f ya filtrado."""
        day0 = pd.Timestamp(day_value).normalize()
        if pd.isna(day0):
            return []
        day1 = day0 + pd.Timedelta(days=1)

        res_day = pd.DataFrame()
        dt_res = pd.Series([], dtype="datetime64[ns]")

        if res_f is not None and not res_f.empty and "Fecha" in res_f.columns:
            res_day = _filter_period(res_f, "day", day0.strftime("%Y-%m-%d"), "Fecha").copy()
            if not res_day.empty:
                fecha = pd.to_datetime(res_day["Fecha"], errors="coerce", dayfirst=True).dt.normalize()
                if "Hora" in res_day.columns:
                    sec = res_day["Hora"].apply(_timeobj_to_seconds).fillna(0)
                    dt_res = fecha + pd.to_timedelta(sec, unit="s")
                else:
                    dt_res = fecha

        rows_out: List[Dict[str, Any]] = []

        for b in (buckets_day or []):
            hh = _crimpado_hour_int_from_bucket(b)
            if hh is None:
                continue

            pn = int(round(float(b.get("cut", 0) or 0)))
            dead_sec = float(b.get("deadSec", 0.0) or 0.0)
            meal_sec = float(b.get("mealSec", 0.0) or 0.0)
            other_sec = float(b.get("otherDeadSec", max(0.0, dead_sec - meal_sec)) or 0.0)

            # No incluir huecos inventados solo para la visualización.
            if pn <= 0 and dead_sec <= 0 and other_sec <= 0 and bool(b.get("_missing")):
                continue
            if pn <= 0 and dead_sec <= 0 and other_sec <= 0:
                continue

            h0 = day0 + pd.Timedelta(hours=int(hh))
            h1 = h0 + pd.Timedelta(hours=1)

            hour_df = pd.DataFrame()
            if res_day is not None and not res_day.empty and len(dt_res) == len(res_day):
                dt_local = pd.to_datetime(dt_res, errors="coerce")
                mask_h = dt_local.notna() & (dt_local >= h0) & (dt_local < h1)
                hour_df = res_day.loc[mask_h].copy()

            meal_sec = max(0.0, min(3600.0, meal_sec))
            other_sec = max(0.0, min(3600.0, other_sec))
            dead_sec, meal_sec, other_sec, _prod_sec = _crimpado_apply_fixed_non_paid_to_parts(
                dead_sec, meal_sec, other_sec, hh, machine=machine, machine_id=machine_id
            )

            # Igual al Excel: 105 reduce TP; las demás paradas son TX.
            tp_h = max(0.0, (3600.0 - meal_sec) / 3600.0)
            tx_h = max(0.0, other_sec / 3600.0)
            tw_h = max(0.0, tp_h - tx_h)

            tc_h_uni = _crimpado_tc_from_resumen_for_hour(
                hour_df,
                tc_lookup=tc_lookup,
                tc_default_h_uni=tc_default_h_uni,
            )

            # Fallback: mantiene compatibilidad si la terminal no está en RESUMEN_TIEMPOS.
            if tc_h_uni in (None, 0) and pn > 0 and tw_h > 0:
                tc_h_uni = tw_h / float(pn)

            tc_h_uni = float(tc_h_uni or 0.0)
            vn = (1.0 / tc_h_uni) if tc_h_uni > 0 else 0.0

            rx = 0
            pb = max(0, pn - rx)
            meta_factor = _crimpado_meta_factor(machine, machine_id)
            meta_plan_time_h = _crimpado_meta_plan_time_h(tp_h, tw_h, machine=machine, machine_id=machine_id)
            pp = (OEE_ESPERADO * vn * meta_plan_time_h * meta_factor) if vn > 0 and meta_plan_time_h > 0 else 0.0
            tcn = (float(pn) * tc_h_uni) if tc_h_uni > 0 else 0.0
            tce = (float(pb) * tc_h_uni) if tc_h_uni > 0 else 0.0

            id_ratio = _safe_div_num(tw_h, tp_h)
            ieo_ratio = _safe_div_num(tcn, tw_h)
            ic_ratio = _safe_div_num(tce, tcn)
            oee_ratio = _safe_div_num(tce, tp_h)
            pa_ratio = _safe_div_num(float(pn), pp)

            rows_out.append({
                "fecha": day0,
                "hour": int(hh),
                "hourLabel": f"{int(hh):02d}:00",
                "referencia": _crimpado_terminales_for_hour(hour_df),
                "operario": _crimpado_operator_for_hour(hour_df, operator=operator),
                "pp": float(pp),
                "pn": int(pn),
                "rx": int(rx),
                "pb": int(pb),
                "tp_h": float(tp_h),
                "tx_h": float(tx_h),
                "tw_h": float(tw_h),
                "tc_h_uni": float(tc_h_uni),
                "vn": float(vn),
                "tcn_h": float(tcn),
                "tce_h": float(tce),
                "id": float(id_ratio),
                "ieo": float(ieo_ratio),
                "ic": float(ic_ratio),
                "oee": float(oee_ratio),
                "pa": float(pa_ratio),
                "meta_factor": float(meta_factor),
                "meta_plan_time_h": float(meta_plan_time_h),
            })

        return rows_out

    def _crimpado_agg_rows(rows_in: List[Dict[str, Any]]) -> Dict[str, float]:
        rows_in = rows_in or []

        pp = float(sum(float(r.get("pp", 0.0) or 0.0) for r in rows_in))
        pn = float(sum(float(r.get("pn", 0.0) or 0.0) for r in rows_in))
        rx = float(sum(float(r.get("rx", 0.0) or 0.0) for r in rows_in))
        pb = float(sum(float(r.get("pb", 0.0) or 0.0) for r in rows_in))
        tp = float(sum(float(r.get("tp_h", 0.0) or 0.0) for r in rows_in))
        tx = float(sum(float(r.get("tx_h", 0.0) or 0.0) for r in rows_in))
        tw = float(sum(float(r.get("tw_h", 0.0) or 0.0) for r in rows_in))
        tcn = float(sum(float(r.get("tcn_h", 0.0) or 0.0) for r in rows_in))
        tce = float(sum(float(r.get("tce_h", 0.0) or 0.0) for r in rows_in))

        # Para dashboard se usa TC promedio ponderado por producción buena.
        # Es más representativo que sumar los TC de cada hora.
        tc_avg = _safe_div_num(tce, pb) if pb > 0 else 0.0
        vn_avg = (1.0 / tc_avg) if tc_avg > 0 else 0.0

        return {
            "pp": pp,
            "pn": pn,
            "rx": rx,
            "pb": pb,
            "tp_h": tp,
            "tx_h": tx,
            "tw_h": tw,
            "tc_h_uni": tc_avg,
            "vn": vn_avg,
            "tcn_h": tcn,
            "tce_h": tce,
            "id": _safe_div_num(tw, tp),
            "ieo": _safe_div_num(tcn, tw),
            "ic": _safe_div_num(tce, tcn),
            "oee": _safe_div_num(tce, tp),
            "pa": _safe_div_num(pn, pp),
        }

    def _crimpado_build_rows_for_current_period() -> List[Dict[str, Any]]:
        p = str(period or "").strip().lower()
        rows_period: List[Dict[str, Any]] = []

        if p == "day":
            day = pd.to_datetime(period_value, errors="coerce")
            if pd.isna(day):
                # fallback por si period_value viene extraño, usa fechas reales filtradas
                days_tmp = _dates_union(res_f, par_f)
            else:
                days_tmp = [pd.Timestamp(day).normalize()]

            for d in days_tmp:
                rows_period.extend(_crimpado_dashboard_rows_for_day(d, hourly_buckets))
            return rows_period

        # Semana/mes: construir buckets por cada día filtrado y sumar sus filas.
        days_tmp = _dates_union(res_f, par_f)
        for d in days_tmp:
            day_str = pd.Timestamp(d).strftime("%Y-%m-%d")
            res_d = _filter_period(res_f, "day", day_str, "Fecha")
            par_d = _filter_period(par_f, "day", day_str, "Fecha")
            buckets_d = _build_hourly_buckets(res_d, par_d)
            rows_period.extend(_crimpado_dashboard_rows_for_day(d, buckets_d))

        return rows_period

    def _crimpado_attach_rows_to_hourly_buckets(rows_in: List[Dict[str, Any]]) -> None:
        if not hourly_buckets or not rows_in:
            return
        row_by_hour = {int(r.get("hour", -1)): r for r in rows_in if r.get("hour") is not None}

        for b in hourly_buckets:
            hh = _crimpado_hour_int_from_bucket(b)
            if hh is None or hh not in row_by_hour:
                continue
            r = row_by_hour[hh]

            # Campos que el JS ya sabe pintar en tarjetas por hora.
            b["plan"] = float(r.get("pp", 0.0) or 0.0)
            b["planned"] = b["plan"]
            b["pp"] = b["plan"]
            b["pn"] = int(r.get("pn", 0) or 0)
            b["rx"] = int(r.get("rx", 0) or 0)
            b["pb"] = int(r.get("pb", 0) or 0)
            b["tpH"] = float(r.get("tp_h", 0.0) or 0.0)
            b["txH"] = float(r.get("tx_h", 0.0) or 0.0)
            b["twH"] = float(r.get("tw_h", 0.0) or 0.0)
            b["tcHUni"] = float(r.get("tc_h_uni", 0.0) or 0.0)
            b["tcSecUni"] = float(r.get("tc_h_uni", 0.0) or 0.0) * 3600.0
            b["vn"] = float(r.get("vn", 0.0) or 0.0)
            b["tcnH"] = float(r.get("tcn_h", 0.0) or 0.0)
            b["tceH"] = float(r.get("tce_h", 0.0) or 0.0)
            b["idPct"] = float(r.get("id", 0.0) or 0.0) * 100.0
            b["ieoPct"] = float(r.get("ieo", 0.0) or 0.0) * 100.0
            b["icPct"] = float(r.get("ic", 0.0) or 0.0) * 100.0
            b["oeePct"] = float(r.get("oee", 0.0) or 0.0) * 100.0
            b["paPct"] = float(r.get("pa", 0.0) or 0.0) * 100.0

            # Compatibilidad con renderProdHour(): usa tcnpSec para TC y capacitySec para PP/PA.
            # En Unión M1, capacitySec queda basado en TW para que las paradas TX reduzcan la meta.
            b["tcnpSec"] = b["tcSecUni"]
            b["capacitySec"] = float(r.get("meta_plan_time_h", r.get("tp_h", 0.0)) or 0.0) * 3600.0
            b["tcnpTotalSec"] = float(r.get("tcn_h", 0.0) or 0.0) * 3600.0
            b["tceTotalSec"] = float(r.get("tce_h", 0.0) or 0.0) * 3600.0
            b["oeeEsperado"] = float(OEE_ESPERADO)
            b["metaFactor"] = float(r.get("meta_factor", _crimpado_meta_factor(machine, machine_id)) or 1.0)
            b["referencia"] = str(r.get("referencia", "") or "")

    def _crimpado_chart_from_rows(rows_in: List[Dict[str, Any]]) -> Dict[str, Any]:
        p = str(period or "").strip().lower()
        rows_in = rows_in or []

        if not rows_in:
            return {"labels": [], "oee": [], "operacional": [], "disponibilidad": [], "calidad": []}

        groups: List[Tuple[str, List[Dict[str, Any]]]] = []

        if p == "day":
            for r in sorted(rows_in, key=lambda x: int(x.get("hour", 0) or 0)):
                groups.append((str(r.get("hourLabel", "") or ""), [r]))

        elif p == "week":
            tmp: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows_in:
                key = pd.Timestamp(r.get("fecha")).strftime("%d/%m")
                tmp.setdefault(key, []).append(r)
            # ordenar por fecha real
            ordered = sorted(tmp.items(), key=lambda kv: pd.to_datetime(kv[1][0].get("fecha"), errors="coerce"))
            groups = [(k, v) for k, v in ordered]

        elif p == "month":
            tmp: Dict[str, List[Dict[str, Any]]] = {}
            sort_key: Dict[str, Tuple[int, int]] = {}
            for r in rows_in:
                f = pd.Timestamp(r.get("fecha"))
                iso = f.isocalendar()
                key = f"Semana {int(iso.week):02d}"
                tmp.setdefault(key, []).append(r)
                sort_key[key] = (int(iso.year), int(iso.week))
            groups = [(k, tmp[k]) for k in sorted(tmp.keys(), key=lambda kk: sort_key.get(kk, (0, 0)))]

        else:
            groups = [("TOTAL", rows_in)]

        labels: List[str] = []
        oee_vals: List[float] = []
        ieo_vals: List[float] = []
        id_vals: List[float] = []
        ic_vals: List[float] = []

        for label, rr in groups:
            agg = _crimpado_agg_rows(rr)
            labels.append(label)
            oee_vals.append(round(float(agg.get("oee", 0.0) or 0.0) * 100.0, 1))
            ieo_vals.append(round(float(agg.get("ieo", 0.0) or 0.0) * 100.0, 1))
            id_vals.append(round(float(agg.get("id", 0.0) or 0.0) * 100.0, 1))
            ic_vals.append(round(float(agg.get("ic", 0.0) or 0.0) * 100.0, 1))

        return {
            "labels": labels,
            "oee": oee_vals,
            "operacional": ieo_vals,
            "disponibilidad": id_vals,
            "calidad": ic_vals,
        }

    crimpado_rows = _crimpado_build_rows_for_current_period()
    crimpado_summary = _crimpado_agg_rows(crimpado_rows)

    if (period or "").strip().lower() == "day" and hourly_buckets:
        _crimpado_attach_rows_to_hourly_buckets(crimpado_rows)

    oee_pct = float(crimpado_summary.get("oee", 0.0) or 0.0) * 100.0
    idx_operacional = float(crimpado_summary.get("ieo", 0.0) or 0.0) * 100.0
    idx_disponibilidad = float(crimpado_summary.get("id", 0.0) or 0.0) * 100.0
    idx_calidad = float(crimpado_summary.get("ic", 0.0) or 0.0) * 100.0
    cumplimiento_plan = float(crimpado_summary.get("pa", 0.0) or 0.0) * 100.0

    # Fallback visual si no hubo filas nominales, para no dejar la disponibilidad vacía.
    if not crimpado_rows:
        idx_disponibilidad = float(pct_eff)

    def _oee_ui_crimpado() -> str:
        return (
            f"{oee_pct:.1f}%||"
            f"Indice de Eficiencia Operacional: {idx_operacional:.1f}% | "
            f"Indice de Disponibilidad: {idx_disponibilidad:.1f}% | "
            f"Indice de Calidad: {idx_calidad:.1f}%"
        )

    oee_chart = _crimpado_chart_from_rows(crimpado_rows)

    tc_avg_h = float(crimpado_summary.get("tc_h_uni", 0.0) or 0.0)
    tc_avg_sec = tc_avg_h * 3600.0
    vn_avg = float(crimpado_summary.get("vn", 0.0) or 0.0)

    kpis_ui: Dict[str, str] = {
        "OEE": _oee_ui_crimpado(),
        "Cumplimiento del plan": f"{cumplimiento_plan:.1f}%",
        "Producción Total": _fmt_int_es(crimpado_summary.get("pn", pulses_total)),

        # Indicadores equivalentes al Excel OEE.
        "Producción Planeada": _fmt_int_es(crimpado_summary.get("pp", 0.0)),
        "Producción Buena": _fmt_int_es(crimpado_summary.get("pb", 0.0)),
        "Producción con Defectos": _fmt_int_es(crimpado_summary.get("rx", 0.0)),
        "Tiempo Pagado": _fmt_hhmm(float(crimpado_summary.get("tp_h", 0.0) or 0.0) * 3600.0),
        "Tiempos Perdidos": (
            f"{_fmt_hhmm(float(crimpado_summary.get('tx_h', 0.0) or 0.0) * 3600.0)}"
            f"||105: {_fmt_hhmm(meal_total_s)} | "
            f"No registrado: {_fmt_hhmm(no_reg_s)} | "
            f"104: {_fmt_hhmm(parada104_s)} | "
            f"{pct_other:.1f}%"
        ),
        "Tiempo Trabajado": (
            f"{_fmt_hhmm(float(crimpado_summary.get('tw_h', 0.0) or 0.0) * 3600.0)} "
            f"({idx_disponibilidad:.1f}%)"
        ),
        "Tiempo de Ciclo": (
            f"{_fmt_dec_es(tc_avg_h, 6)} h/uni||"
            f"Segundos: {_fmt_dec_es(tc_avg_sec, 2)} s/uni"
        ),
        "Velocidad Nominal": _fmt_units_per_hour(vn_avg),
        "Tiempo Normal (TCN)": _fmt_hhmm(float(crimpado_summary.get("tcn_h", 0.0) or 0.0) * 3600.0),
        "Tiempo Estándar (TCE)": _fmt_hhmm(float(crimpado_summary.get("tce_h", 0.0) or 0.0) * 3600.0),
        "Índice de Disponibilidad": f"{idx_disponibilidad:.1f}%",
        "Índice de Eficiencia Operacional": f"{idx_operacional:.1f}%",
        "Índice de Calidad": f"{idx_calidad:.1f}%",
        "PA": f"{cumplimiento_plan:.1f}%",
        "Crimpados Carrete": f"{_fmt_int_es(carrete)} ({_fmt_pct_es(pct_carrete)})",
        "Crimpados Manual": f"{_fmt_int_es(manual)} ({_fmt_pct_es(pct_manual)})",
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
        "other_causes": other_causes,
        "kpis": {
            "hourly_buckets": hourly_buckets,
            "other_causes": other_causes,
            "crimpado_summary": crimpado_summary,
            "crimpado_rows": crimpado_rows,
            "tc_nominal_history_h_uni": float(tc_nominal_history_h_uni or 0.0),
            "tc_nominal_history_sec": float(tc_nominal_history_h_uni or 0.0) * 3600.0,
            "oee_esperado": float(OEE_ESPERADO),
            "oee_esperado_pct": float(OEE_ESPERADO) * 100.0,
            "meta_factor": float(_crimpado_meta_factor(machine, machine_id)),
        },
        "terminal_usage": terminal_usage,
        "oee_chart": oee_chart,
        "machine_id": _norm_machine_id(machine_id),
    }


# ============================================================
# EXPORT EXCEL CRIMPADO / APLICACIÓN / UNIÓN
# Modelo tipo THB adaptado a terminales aplicadas
# ============================================================
def _clean_export_terminal(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null", "nat"):
        return ""
    s = re.sub(r"\.0$", "", s)
    return s.strip()


def _format_crimpado_equipo(machine: str, machine_id: str = "") -> str:
    m = (machine or "APLICACION").strip().upper()
    mid = _norm_machine_id(machine_id)
    if mid:
        return f"{m} M{mid}"
    return m


def _app_export_date_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Fecha" not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")
    s = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    return s.dropna().dt.normalize()


def _app_days_for_period_export(
    res_df: pd.DataFrame,
    par_df: pd.DataFrame,
    period: str,
    period_value: str,
    operator: str = "",
) -> List[pd.Timestamp]:
    op = (operator or "").strip()
    if op.lower() == "general":
        op = ""

    res_f = _filter_period(res_df, period, period_value, "Fecha") if res_df is not None else pd.DataFrame()
    par_f = _filter_period(par_df, period, period_value, "Fecha") if par_df is not None else pd.DataFrame()

    if op:
        if res_f is not None and not res_f.empty and "Nombre" in res_f.columns:
            res_f = res_f[res_f["Nombre"].astype(str).str.strip().eq(op)]
        if par_f is not None and not par_f.empty and "Nombre" in par_f.columns:
            par_f = par_f[par_f["Nombre"].astype(str).str.strip().eq(op)]

    d1 = _app_export_date_series(res_f)
    d2 = _app_export_date_series(par_f)
    all_days = pd.concat([d1, d2], ignore_index=True).dropna()
    if all_days.empty:
        return []

    days = pd.to_datetime(all_days.unique(), errors="coerce")
    days = days[~pd.isna(days)]
    return [pd.Timestamp(d).normalize() for d in sorted(days)]


def _crimpado_terminales_for_hour(hour_df: pd.DataFrame) -> str:
    if hour_df is None or hour_df.empty or "Terminal" not in hour_df.columns:
        return ""

    vistos = set()
    terminales: List[str] = []
    vals = hour_df["Terminal"].dropna().astype(str).tolist()
    for raw in vals:
        t = _clean_export_terminal(raw)
        if not t:
            continue
        k = t.upper()
        if k in vistos:
            continue
        vistos.add(k)
        terminales.append(t)

    return ", ".join(terminales)


def _crimpado_operator_for_hour(hour_df: pd.DataFrame, operator: str = "") -> str:
    op = (operator or "").strip()
    if op and op.lower() != "general":
        return op

    if hour_df is not None and not hour_df.empty and "Nombre" in hour_df.columns:
        ops = hour_df["Nombre"].dropna().astype(str).str.strip()
        ops = ops[ops.ne("") & ~ops.str.lower().isin(["nan", "none", "general"])]
        if not ops.empty:
            try:
                return str(ops.mode().iloc[0]).strip()
            except Exception:
                return str(ops.iloc[0]).strip()

    return "General"


def _crimpado_norm_col_name(c: Any) -> str:
    """Normaliza encabezados con saltos de línea, tildes y espacios."""
    s = str(c or "").replace("\n", " ").replace("\r", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    trans = str.maketrans("áéíóúüñ", "aeiouun")
    return s.translate(trans)


def _crimpado_find_col(df: pd.DataFrame, required_tokens: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    req = [_crimpado_norm_col_name(x) for x in required_tokens]
    for c in df.columns:
        nc = _crimpado_norm_col_name(c)
        if all(tok in nc for tok in req):
            return c
    return None


def _crimpado_num_float(x: Any) -> Optional[float]:
    """Convierte números tipo 20.6 / '20,6' / '1.234,56' a float."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return float(x)

    s = str(x).strip().replace(" ", "")
    if not s or s.lower() in ("nan", "none", "null"):
        return None

    lc = s.rfind(",")
    ld = s.rfind(".")
    if lc > -1 and ld > -1:
        if lc > ld:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif lc > -1:
        s = s.replace(".", "").replace(",", ".")

    try:
        return float(s)
    except Exception:
        return None


def _crimpado_terminal_key(x: Any) -> str:
    """
    Llave robusta para cruzar Terminal entre:
    - RESUMEN_TIEMPOS
    - APLICACION1_Resumen_Pulsos
    - Referencia exportada
    """
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return ""
    if isinstance(x, (int, float, np.integer, np.floating)):
        xf = float(x)
        if np.isfinite(xf) and abs(xf - round(xf)) < 1e-9:
            return str(int(round(xf)))
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null", "nat"):
        return ""
    s = re.sub(r"\.0$", "", s)
    return s.strip().upper()


def load_resumen_tiempos(
    path: str,
    machine: str = "APLICACION",
    machine_id: str = "",
) -> pd.DataFrame:
    """
    Lee la hoja RESUMEN_TIEMPOS.

    Estructura esperada:
      Terminal | Sesiones registradas | Tiempo Neto Total | Unidades Aplicadas |
      Promedio (segundos) | Promedio (mm:ss)

    Nota:
    - Para Google Sheets con GSHEET_APP_TOKEN se lee por export XLSX, porque esta
      hoja nueva no necesita un GID adicional en config.py.
    - Si la hoja no existe, retorna DataFrame vacío y el export conserva fallback.
    """
    sheet_id, _gid_res, _gid_par, _ttl_s = _pick_gsheet_params(machine, machine_id)
    m = (machine or "APLICACION").strip().upper()
    mid = (machine_id or "").strip()

    prefix = f"{m}{mid}" if mid else m
    sheet_candidates = [
        f"{prefix}_RESUMEN_TIEMPOS",
        f"{prefix}_Resumen_Tiempos",
        f"{prefix} RESUMEN_TIEMPOS",
        "RESUMEN_TIEMPOS",
        "Resumen_Tiempos",
        "Resumen Tiempos",
    ]

    try:
        if not path:
            raw = _read_sheet_try(GSHEET_APP_TOKEN, sheet_candidates, header=None, sheet_id=sheet_id)
        else:
            token_sheet_id = sheet_id if path == GSHEET_APP_TOKEN else ""
            raw = _read_sheet_try(path, sheet_candidates, header=None, sheet_id=token_sheet_id)
    except Exception:
        return pd.DataFrame()

    try:
        hi = _find_header_row(raw, {"Terminal"}, max_scan=25)
        header = raw.iloc[hi].astype(str).str.strip().tolist()
        df = raw.iloc[hi + 1 :].copy()
        df.columns = header
        df = df.dropna(how="all").reset_index(drop=True)

        # Quitar filas de notas/metodología que vienen debajo de la tabla.
        terminal_col = _crimpado_find_col(df, ["terminal"])
        if not terminal_col:
            return pd.DataFrame()

        df = df[df[terminal_col].notna()].copy()
        df = df[~df[terminal_col].astype(str).str.strip().str.lower().isin(["", "nan", "none"])].copy()
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _crimpado_build_tc_lookup(
    resumen_tiempos_df: pd.DataFrame,
) -> Tuple[Dict[str, float], Optional[float]]:
    """
    Convierte RESUMEN_TIEMPOS a:
      - lookup[terminal] = tiempo de ciclo en horas decimales (h/uni)
      - promedio_global = fallback en h/uni si existe la fila PROMEDIO GLOBAL
    """
    if resumen_tiempos_df is None or resumen_tiempos_df.empty:
        return {}, None

    df = resumen_tiempos_df.copy()

    terminal_col = _crimpado_find_col(df, ["terminal"])
    prom_sec_col = _crimpado_find_col(df, ["promedio", "seg"])
    if not terminal_col or not prom_sec_col:
        return {}, None

    lookup: Dict[str, float] = {}
    promedio_global: Optional[float] = None

    for _, r in df.iterrows():
        raw_terminal = r.get(terminal_col)
        key = _crimpado_terminal_key(raw_terminal)
        sec = _crimpado_num_float(r.get(prom_sec_col))

        if sec is None or sec <= 0:
            continue

        tc_h = float(sec) / 3600.0

        raw_txt = str(raw_terminal or "").strip().upper()
        if "PROMEDIO" in raw_txt and "GLOBAL" in raw_txt:
            promedio_global = tc_h
            continue

        if key:
            lookup[key] = tc_h

    return lookup, promedio_global


def _crimpado_tc_from_resumen_for_hour(
    hour_df: pd.DataFrame,
    tc_lookup: Optional[Dict[str, float]] = None,
    tc_default_h_uni: Optional[float] = None,
) -> Optional[float]:
    """
    Retorna TC en horas decimales usando RESUMEN_TIEMPOS.

    Si en una misma hora hay varias terminales, calcula promedio ponderado:
      TC_hora = SUM(Pulsos_terminal * TC_terminal) / SUM(Pulsos_terminal)

    Si una terminal no existe en RESUMEN_TIEMPOS, usa PROMEDIO GLOBAL si está
    disponible. Si no hay ninguna coincidencia, retorna None.
    """
    tc_lookup = tc_lookup or {}
    if hour_df is None or hour_df.empty or "Terminal" not in hour_df.columns:
        return tc_default_h_uni if tc_default_h_uni and tc_default_h_uni > 0 else None

    weighted_sum = 0.0
    weight_sum = 0.0

    for _, r in hour_df.iterrows():
        key = _crimpado_terminal_key(r.get("Terminal"))
        if not key:
            continue

        tc = tc_lookup.get(key)
        if tc is None or tc <= 0:
            tc = tc_default_h_uni

        if tc is None or tc <= 0:
            continue

        w = _crimpado_num_float(r.get("Pulsos", 1))
        if w is None or w <= 0:
            w = 1.0

        weighted_sum += float(tc) * float(w)
        weight_sum += float(w)

    if weight_sum > 0:
        return weighted_sum / weight_sum

    return tc_default_h_uni if tc_default_h_uni and tc_default_h_uni > 0 else None


def _crimpado_is_union1(machine: str = "", machine_id: str = "") -> bool:
    """Modelo nominal fijo solicitado únicamente para Unión - Máquina 1."""
    return (str(machine or "").strip().upper() == "UNION") and (str(machine_id or "").strip() == "1")


def _crimpado_fixed_non_paid_sec_for_hour(hour: Any, machine: str = "", machine_id: str = "") -> float:
    """
    Tiempo no pago fijo para Unión 1.
    Se trata como comida/tiempo no pago, no como TX:
      - 08:00-09:00 -> 15 min desayuno
      - 13:00-14:00 -> 30 min almuerzo
    """
    if not _crimpado_is_union1(machine, machine_id):
        return 0.0
    try:
        h = int(hour)
    except Exception:
        return 0.0
    if h == 8:
        return 15.0 * 60.0
    if h == 13:
        return 30.0 * 60.0
    return 0.0


def _crimpado_apply_fixed_non_paid_to_parts(
    dead_sec: Any,
    meal_sec: Any,
    other_sec: Any,
    hour: Any,
    machine: str = "",
    machine_id: str = "",
) -> Tuple[float, float, float, float]:
    """
    Ajusta una hora para garantizar el desayuno/almuerzo no pago de Unión 1.
    Retorna (dead_sec, meal_sec, other_sec, prod_sec).

    La regla evita duplicar la parada 105: si ya hay 105 registrada, se toma
    el mayor valor entre lo registrado y el fijo.
    """
    dead = max(0.0, min(3600.0, float(dead_sec or 0.0)))
    meal = max(0.0, min(3600.0, float(meal_sec or 0.0)))
    other = max(0.0, min(3600.0, float(other_sec or 0.0)))

    if other <= 0 and dead > meal:
        other = max(0.0, dead - meal)

    fixed_meal = _crimpado_fixed_non_paid_sec_for_hour(hour, machine, machine_id)
    if fixed_meal > 0:
        meal = max(meal, fixed_meal)

    dead = max(dead, meal + other)
    dead = min(3600.0, dead)
    if meal > dead:
        meal = dead
    other = max(0.0, min(3600.0 - meal, other))
    if meal + other > dead:
        other = max(0.0, dead - meal)

    prod = max(0.0, 3600.0 - dead)
    return float(dead), float(meal), float(other), float(prod)


def _crimpado_paradas_with_downtime(par_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega/normaliza downtime_s en una hoja de paradas de crimpado."""
    if par_df is None or par_df.empty:
        return pd.DataFrame()

    df = par_df.copy()
    if "downtime_s" in df.columns:
        df["downtime_s"] = pd.to_numeric(df["downtime_s"], errors="coerce").fillna(0.0).astype(float)
    else:
        dur_raw = df.get("Duracion", pd.Series([None] * len(df), index=df.index, dtype="object"))
        dt = pd.to_timedelta(dur_raw, errors="coerce")
        secs = dt.dt.total_seconds()

        if secs.isna().all() and ("Hora de Inicio" in df.columns) and ("Hora de Fin" in df.columns):
            ini = df["Hora de Inicio"].apply(_timeobj_to_seconds)
            fin = df["Hora de Fin"].apply(_timeobj_to_seconds)
            ini = pd.to_numeric(ini, errors="coerce")
            fin = pd.to_numeric(fin, errors="coerce")
            diff = fin - ini
            diff = np.where(diff < 0, diff + 86400, diff)
            secs = pd.Series(diff, index=df.index, dtype="float64")

        df["downtime_s"] = pd.to_numeric(secs, errors="coerce").fillna(0.0).astype(float)

    df["CausaCode"] = pd.to_numeric(df.get("Causa", np.nan), errors="coerce")
    return df


def _crimpado_estimate_tc_nominal_from_history(
    res_df: pd.DataFrame,
    par_df: pd.DataFrame,
    machine: str = "APLICACION",
    machine_id: str = "",
    operator: str = "",
) -> Optional[float]:
    """
    Calcula el tiempo estándar nominal para Unión 1 cuando no existe RESUMEN_TIEMPOS.

    Modelo solicitado para una sola terminal:
      TC_nom_UNION1 = SUM(TW_h) / SUM(PN_h)
      TW_h = 3600 - TNP_h - TX_h

    Retorna horas/unidad. Internamente calcula segundos/terminal y divide entre 3600.
    """
    if not _crimpado_is_union1(machine, machine_id):
        return None
    if res_df is None or res_df.empty or "Pulsos" not in res_df.columns:
        return None
    if "Fecha" not in res_df.columns or "Hora" not in res_df.columns:
        return None

    op = str(operator or "").strip()
    res = res_df.copy()
    if op and op.lower() != "general" and "Nombre" in res.columns:
        res = res[res["Nombre"].astype(str).str.strip().eq(op)].copy()

    if res.empty:
        return None

    tmp = res.dropna(subset=["Fecha", "Hora"]).copy()
    tmp["Fecha_d"] = pd.to_datetime(tmp["Fecha"], errors="coerce", dayfirst=True).dt.date
    tmp["Hora_h"] = tmp["Hora"].apply(lambda t: int(t.hour) if hasattr(t, "hour") else np.nan)
    tmp["Pulsos"] = pd.to_numeric(tmp["Pulsos"], errors="coerce").fillna(0).astype(float)
    tmp = tmp.dropna(subset=["Fecha_d", "Hora_h"]).copy()
    tmp["Hora_h"] = tmp["Hora_h"].astype(int)
    tmp = tmp[tmp["Pulsos"] > 0].copy()
    if tmp.empty:
        return None

    pulses_by_slot = tmp.groupby(["Fecha_d", "Hora_h"], dropna=False)["Pulsos"].sum().to_dict()

    downtime_by_slot: Dict[Tuple[Any, int], float] = {}
    meal_by_slot: Dict[Tuple[Any, int], float] = {}

    ptmp = _crimpado_paradas_with_downtime(par_df)
    if ptmp is not None and not ptmp.empty and "Fecha" in ptmp.columns and "Hora de Inicio" in ptmp.columns:
        if op and op.lower() != "general" and "Nombre" in ptmp.columns:
            ptmp = ptmp[ptmp["Nombre"].astype(str).str.strip().eq(op)].copy()

        if "Tipo de Intervalo" in ptmp.columns:
            ptmp = ptmp[~ptmp["Tipo de Intervalo"].astype(str).str.contains("marcador", case=False, na=False)].copy()

        ptmp["Fecha0"] = pd.to_datetime(ptmp["Fecha"], errors="coerce", dayfirst=True).dt.normalize()
        sec0 = ptmp["Hora de Inicio"].apply(_timeobj_to_seconds)
        sec0 = pd.to_numeric(sec0, errors="coerce").fillna(0).astype(int)
        start_dt = ptmp["Fecha0"] + pd.to_timedelta(sec0, unit="s")
        dur_s = pd.to_numeric(ptmp.get("downtime_s", 0.0), errors="coerce").fillna(0.0).astype(float)
        end_dt = start_dt + pd.to_timedelta(dur_s, unit="s")

        for s0, e0, cc in zip(start_dt.tolist(), end_dt.tolist(), ptmp.get("CausaCode", pd.Series([np.nan] * len(ptmp))).tolist()):
            if pd.isna(s0) or pd.isna(e0) or e0 <= s0:
                continue
            try:
                code_int = int(cc) if not pd.isna(cc) else None
            except Exception:
                code_int = None

            cur = s0
            while cur < e0:
                hour_start = cur.replace(minute=0, second=0, microsecond=0)
                next_hour = hour_start + pd.Timedelta(hours=1)
                overlap_end = e0 if e0 < next_hour else next_hour
                secs = float((overlap_end - cur).total_seconds())
                slot = (hour_start.date(), int(hour_start.hour))
                downtime_by_slot[slot] = downtime_by_slot.get(slot, 0.0) + secs
                if code_int == 105:
                    meal_by_slot[slot] = meal_by_slot.get(slot, 0.0) + secs
                cur = overlap_end

    total_tw_s = 0.0
    total_pn = 0.0

    for (d, h), pn in pulses_by_slot.items():
        pn = float(pn or 0.0)
        if pn <= 0:
            continue

        dead = float(downtime_by_slot.get((d, h), 0.0) or 0.0)
        meal = float(meal_by_slot.get((d, h), 0.0) or 0.0)
        other = max(0.0, dead - meal)
        _dead, _meal, other, prod = _crimpado_apply_fixed_non_paid_to_parts(
            dead, meal, other, h, machine=machine, machine_id=machine_id
        )
        tw_s = max(0.0, prod)

        # Filtro suave para evitar horas corruptas o extremadamente pequeñas.
        if tw_s <= 0 or pn < 5:
            continue

        tc_s = tw_s / pn
        if not (1.0 <= tc_s <= 180.0):
            continue

        total_tw_s += tw_s
        total_pn += pn

    if total_tw_s <= 0 or total_pn <= 0:
        return None

    tc_sec = total_tw_s / total_pn
    if tc_sec <= 0 or not np.isfinite(tc_sec):
        return None
    return float(tc_sec) / 3600.0


def _crimpado_hour_int_from_bucket(b: Dict[str, Any]) -> Optional[int]:
    raw = str(b.get("hourLabel") or b.get("hour") or "").strip()
    m = re.search(r"(\d{1,2})", raw)
    if not m:
        return None
    try:
        hh = int(m.group(1))
        if 0 <= hh <= 23:
            return hh
    except Exception:
        return None
    return None


def _crimpado_rows_for_day(
    res_df: pd.DataFrame,
    day: pd.Timestamp,
    buckets: List[Dict[str, Any]],
    machine: str = "APLICACION",
    machine_id: str = "",
    operator: str = "",
    tc_lookup: Optional[Dict[str, float]] = None,
    tc_default_h_uni: Optional[float] = None,
) -> List[Dict[str, Any]]:
    day0 = pd.Timestamp(day).normalize()
    day1 = day0 + pd.Timedelta(days=1)

    res_day = pd.DataFrame()
    dt_res = pd.Series([], dtype="datetime64[ns]")

    if res_df is not None and not res_df.empty and "Fecha" in res_df.columns:
        res_day = _filter_period(res_df, "day", day0.strftime("%Y-%m-%d"), "Fecha").copy()
        op = (operator or "").strip()
        if op and op.lower() != "general" and "Nombre" in res_day.columns:
            res_day = res_day[res_day["Nombre"].astype(str).str.strip().eq(op)].copy()

        if not res_day.empty:
            fecha = pd.to_datetime(res_day["Fecha"], errors="coerce", dayfirst=True).dt.normalize()
            if "Hora" in res_day.columns:
                sec = res_day["Hora"].apply(_timeobj_to_seconds).fillna(0)
                dt_res = fecha + pd.to_timedelta(sec, unit="s")
            else:
                dt_res = fecha

    rows: List[Dict[str, Any]] = []
    equipo = _format_crimpado_equipo(machine, machine_id)

    for b in (buckets or []):
        hh = _crimpado_hour_int_from_bucket(b)
        if hh is None:
            continue

        pn = int(round(float(b.get("cut", 0) or 0)))
        dead_sec = float(b.get("deadSec", 0.0) or 0.0)
        meal_sec = float(b.get("mealSec", 0.0) or 0.0)
        other_sec = float(b.get("otherDeadSec", max(0.0, dead_sec - meal_sec)) or 0.0)

        # No exportar huecos creados solamente para la visualización.
        if pn <= 0 and dead_sec <= 0 and other_sec <= 0 and bool(b.get("_missing")):
            continue
        if pn <= 0 and dead_sec <= 0 and other_sec <= 0:
            continue

        h0 = day0 + pd.Timedelta(hours=int(hh))
        h1 = h0 + pd.Timedelta(hours=1)

        hour_df = pd.DataFrame()
        if res_day is not None and not res_day.empty and len(dt_res) == len(res_day):
            dt_local = pd.to_datetime(dt_res, errors="coerce")
            mask_h = dt_local.notna() & (dt_local >= h0) & (dt_local < h1)
            hour_df = res_day.loc[mask_h].copy()

        referencia = _crimpado_terminales_for_hour(hour_df)
        operario = _crimpado_operator_for_hour(hour_df, operator=operator)

        meal_sec = max(0.0, min(3600.0, meal_sec))
        other_sec = max(0.0, min(3600.0, other_sec))
        dead_sec, meal_sec, other_sec, _prod_sec = _crimpado_apply_fixed_non_paid_to_parts(
            dead_sec, meal_sec, other_sec, hh, machine=machine, machine_id=machine_id
        )

        # En crimpado la parada 105 no se mete a TX; reduce el tiempo pagado.
        tp_h = max(0.0, (3600.0 - meal_sec) / 3600.0)
        tx_h = max(0.0, other_sec / 3600.0)
        tw_h = max(0.0, tp_h - tx_h)

        # TC = tiempo de ciclo nominal por terminal desde RESUMEN_TIEMPOS.
        # La hoja trae Promedio (segundos); se convierte a horas decimales (h/uni).
        # Si hay varias terminales en la misma hora, se usa promedio ponderado por Pulsos.
        tc_h_uni = _crimpado_tc_from_resumen_for_hour(
            hour_df,
            tc_lookup=tc_lookup,
            tc_default_h_uni=tc_default_h_uni,
        )

        # Fallback de seguridad: si no existe RESUMEN_TIEMPOS o no hay coincidencia,
        # conserva el cálculo anterior para no romper la descarga.
        if tc_h_uni in (None, 0) and pn > 0 and tw_h > 0:
            tc_h_uni = tw_h / float(pn)

        meta_factor = _crimpado_meta_factor(machine, machine_id)
        meta_plan_time_h = _crimpado_meta_plan_time_h(tp_h, tw_h, machine=machine, machine_id=machine_id)

        rows.append({
            "fecha": day0.to_pydatetime(),
            "referencia": referencia,
            "equipo": equipo,
            "operario": operario,
            "inicio": h0.to_pydatetime(),
            "fin": h1.to_pydatetime(),
            "pp": None,
            "pn": int(pn),
            "rx": 0,
            "tp_h": round(float(tp_h), 3),
            "tx_h": round(float(tx_h), 3),
            "tw_h": round(float(tw_h), 3),
            "tc_h_uni": tc_h_uni,
            "meta_factor": meta_factor,
            "meta_plan_time_h": round(float(meta_plan_time_h), 3),
        })

    return rows


def get_crimpado_export_day_exports(
    path: str,
    period: str,
    period_value: str,
    machine: str = "APLICACION",
    subcat: str = "",
    operator: str = "",
    machine_id: str = "",
) -> List[Dict[str, Any]]:
    """
    Construye los datos para descargar Excel de crimpado usando el mismo
    modelo de THB, pero con:
      - Referencia = terminal(es) aplicada(s)
      - Equipo = APLICACION/UNION + M1/M2
      - PN = pulsos / terminales aplicadas
      - RX = 0 si no hay datos de defectos
      - TC = Promedio(segundos) / 3600 desde RESUMEN_TIEMPOS, cruzado por Terminal.
    """
    m = (machine or "APLICACION").strip().upper()
    mid = _norm_machine_id(machine_id)
    op = (operator or "").strip()
    if op.lower() == "general":
        op = ""

    res_df = load_resumen_pulsos(path, machine=m, machine_id=mid)
    par_df, _ = load_paradas(path, machine=m, machine_id=mid)

    # Hoja nueva: RESUMEN_TIEMPOS.
    # De aquí sale el TC exportado en la columna N del Excel OEE.
    resumen_tiempos_df = load_resumen_tiempos(path, machine=m, machine_id=mid)
    tc_lookup, tc_default_h_uni = _crimpado_build_tc_lookup(resumen_tiempos_df)

    # Si no hay RESUMEN_TIEMPOS, Unión 1 usa un TC nominal histórico constante:
    # TC_nom_UNION1 = SUM(TW_h) / SUM(PN_h).
    tc_nominal_history_h_uni = _crimpado_estimate_tc_nominal_from_history(
        res_df=res_df,
        par_df=par_df,
        machine=m,
        machine_id=mid,
        operator=op,
    )
    if (tc_default_h_uni is None or float(tc_default_h_uni or 0.0) <= 0.0) and tc_nominal_history_h_uni:
        tc_default_h_uni = float(tc_nominal_history_h_uni)

    days = _app_days_for_period_export(res_df, par_df, period, period_value, operator=op)
    if not days:
        return []

    day_exports: List[Dict[str, Any]] = []
    for day in days:
        day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
        result_day = run_aplicacion_analysis(
            path=path,
            period="day",
            period_value=day_str,
            machine=m,
            subcat=subcat,
            operator=op,
            time_start="",
            time_end="",
            machine_id=mid,
        )

        buckets = []
        if isinstance(result_day, dict):
            buckets = result_day.get("kpis", {}).get("hourly_buckets", []) or []

        rows = _crimpado_rows_for_day(
            res_df=res_df,
            day=pd.Timestamp(day),
            buckets=buckets,
            machine=m,
            machine_id=mid,
            operator=op,
            tc_lookup=tc_lookup,
            tc_default_h_uni=tc_default_h_uni,
        )

        if not rows:
            continue

        day_exports.append({
            "sheet_title": day_str,
            "rows": rows,
            "oee_expected": _crimpado_oee_esperado(m, mid),
            "meta_factor": _crimpado_meta_factor(m, mid),
        })

    return day_exports


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

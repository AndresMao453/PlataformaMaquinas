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

    # --- FIX: Hora puede venir float/NaN (Excel/Sheets). Convertir a texto HH:MM ---
    def _coerce_time_str(x):
        if pd.isna(x):
            return ""
        # Excel suele guardar hora como fracción del día (0..1): 0.5 => 12:00
        if isinstance(x, (int, float, np.integer, np.floating)):
            fx = float(x)
            if 0 <= fx < 1.1:
                secs = int(round(fx * 24 * 3600))
                hh = (secs // 3600) % 24
                mm = (secs % 3600) // 60
                return f"{hh:02d}:{mm:02d}"
            # A veces viene como HHMM (ej: 830, 1630)
            if fx.is_integer():
                n = int(fx)
                if 0 <= n <= 2359:
                    hh = n // 100
                    mm = n % 100
                    if 0 <= hh <= 23 and 0 <= mm <= 59:
                        return f"{hh:02d}:{mm:02d}"
            return str(x).strip()
        return str(x).strip()

    ss = ss.map(_coerce_time_str)

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

    if not path:
        gs = load_gsheet_tab_csv(sheet_id, int(gid_res), ttl_s=ttl_s)
        raw = pd.concat([pd.DataFrame([gs.columns.tolist()]), gs.reset_index(drop=True)], ignore_index=True)
    else:
        sheet_candidates = [
            # ✅ Unión
            "UNION2_Resumen_Pulsos",
            "Union2_Resumen_Pulsos",
            "UNION1_Resumen_Pulsos",
            "Union1_Resumen_Pulsos",
            "UNION_Resumen_Pulsos",
            "Resumen_Pulsos_UNION",

            # ✅ Aplicación
            "APLICACION2_Resumen_Pulsos",
            "Aplicacion2_Resumen_Pulsos",
            "APLICACION1_Resumen_Pulsos",
            "Aplicacion1_Resumen_Pulsos",
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

    if not path:
        gs = load_gsheet_tab_csv(sheet_id, int(gid_par), ttl_s=ttl_s)
        raw = pd.concat([pd.DataFrame([gs.columns.tolist()]), gs.reset_index(drop=True)], ignore_index=True)
    else:
        sheet_candidates = [
            "Paradas",  # ✅ nombre real

            # compat anteriores
            "UNION2_Paradas",
            "UNION1_Paradas",
            "UNION_Paradas",
            "APLICACION2_Paradas",
            "Aplicacion2_Paradas",
            "APLICACION1_Paradas",
            "Aplicacion1_Paradas",
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

    for j, val in enumerate(first_row):
        if str(val).upper() == "CODIGO":
            code_col = df.columns[j]
            if j + 1 < len(df.columns):
                desc_col = df.columns[j + 1]
            break

    mapping: Dict[int, str] = {}
    if code_col and desc_col and code_col in df.columns and desc_col in df.columns:
        codes = df[code_col].astype(str)
        descs = df[desc_col].astype(str)
        for c, d in zip(codes, descs):
            m = re.match(r"^\s*(\d+)", c)
            if m:
                desc = d.strip()
                mapping[int(m.group(1))] = "" if desc.lower() == "nan" else desc

    keep = [
        c
        for c in [
            "Fecha",
            "Nombre",
            "Hora de Inicio",
            "Hora de Fin",
            "Duración [hh]:mm:ss",
            "Tipo de Intervalo",
            "Causa",
            "Conteo",
            "Terminal",
            "Aplicación",
            "Tipo de Reproceso",
        ]
        if c in df.columns
    ]
    df = df[keep].copy()

    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    else:
        df["Fecha"] = pd.NaT

    for col in ["Hora de Inicio", "Hora de Fin"]:
        if col in df.columns:
            df[col] = _parse_excel_time_series(df[col])

    def parse_duration(x: Any):
        if pd.isna(x):
            return pd.NaT
        if isinstance(x, pd.Timedelta):
            return x
        try:
            return pd.to_timedelta(str(x))
        except Exception:
            return pd.NaT

    if "Duración [hh]:mm:ss" in df.columns:
        df["Duracion"] = df["Duración [hh]:mm:ss"].apply(parse_duration)
    else:
        df["Duracion"] = pd.NaT

    if "Tipo de Intervalo" in df.columns:
        df["Tipo de Intervalo"] = df["Tipo de Intervalo"].astype(str)
    else:
        df["Tipo de Intervalo"] = pd.Series([""] * len(df), index=df.index, dtype="object")

    if "Causa" in df.columns:
        df["CausaCode"] = pd.to_numeric(df["Causa"], errors="coerce")
    else:
        df["CausaCode"] = np.nan

    df["CausaDesc"] = df["CausaCode"].map(mapping)

    if "Conteo" in df.columns:
        df["Conteo"] = pd.to_numeric(df["Conteo"], errors="coerce").fillna(0)
    else:
        df["Conteo"] = 0

    if "Nombre" in df.columns:
        df["Nombre"] = df["Nombre"].astype(str)
    else:
        df["Nombre"] = ""

    if "Terminal" in df.columns:
        df["Terminal"] = df["Terminal"].astype(str)
    if "Aplicación" in df.columns:
        df["Aplicación"] = df["Aplicación"].astype(str)

    return df, mapping


# ============================================================
# Periodos disponibles
# ============================================================
def _dates_union(res_df: pd.DataFrame, par_df: pd.DataFrame) -> List[pd.Timestamp]:
    d1 = res_df["Fecha"].dropna().dt.normalize() if "Fecha" in res_df.columns else pd.Series([], dtype="datetime64[ns]")
    d2 = par_df["Fecha"].dropna().dt.normalize() if "Fecha" in par_df.columns else pd.Series([], dtype="datetime64[ns]")
    dates = pd.concat([d1, d2]).dropna().unique()
    dates = pd.to_datetime(dates)
    return sorted(dates)


def get_period_options_aplicacion(
    path: str,
    period: str,
    machine: str = "APLICACION",
    machine_id: str = "",
) -> List[Dict[str, str]]:
    res_df = load_resumen_pulsos(path, machine=machine, machine_id=machine_id)
    par_df, _ = load_paradas(path, machine=machine, machine_id=machine_id)

    dates = _dates_union(res_df, par_df)
    if not dates:
        return []

    dates = [d.date() for d in dates]

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

    s = pd.to_datetime(df[date_col], errors="coerce")

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

        # ✅ FIX: no convertir a int (evita "cannot convert NA to integer")
        iso = s.dt.isocalendar()
        mask = (iso["year"] == y) & (iso["week"] == w)

        # mask puede traer <NA> cuando Fecha es NaT
        return df[mask.fillna(False)]


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
        if res_f is None or res_f.empty:
            return []
        if "Fecha" not in res_f.columns or "Hora" not in res_f.columns:
            return []

        tmp = res_f.dropna(subset=["Fecha", "Hora"]).copy()
        tmp["Fecha_d"] = pd.to_datetime(tmp["Fecha"], errors="coerce").dt.date
        tmp["Hora_h"] = tmp["Hora"].apply(lambda t: int(t.hour) if hasattr(t, "hour") else np.nan)
        tmp["Hora_h"] = pd.to_numeric(tmp["Hora_h"], errors="coerce")
        tmp = tmp.dropna(subset=["Fecha_d", "Hora_h"]).copy()
        tmp["Hora_h"] = tmp["Hora_h"].astype(int)

        if tmp.empty:
            return []

        tmp["Pulsos"] = pd.to_numeric(tmp.get("Pulsos", 0), errors="coerce").fillna(0).astype(int)
        pulses_by_slot = tmp.groupby(["Fecha_d", "Hora_h"])["Pulsos"].sum().to_dict()
        active_slots = sorted(pulses_by_slot.keys(), key=lambda x: (x[0], x[1]))

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
            # finalizar -> inicio
            if interval_key == "finalizar":
                return "finalizar→inicio"
            # reproceso_inicio -> reproceso_fin, prueba_tension_inicio -> prueba_tension_fin, etc
            if interval_key.endswith("_inicio"):
                base = interval_key[:-6]  # quita "_inicio"
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

                    # ✅ clave final:
                    # - si hay código: "203"
                    # - si no hay código: intervalos especiales (finalizar→inicio, reproceso_inicio→reproceso_fin, etc)
                    interval_key = _norm_interval_key(intervalo)
                    pair_key = _interval_pair_key(interval_key)

                    if code_int is not None:
                        key_code = str(code_int)
                        key_desc = (desc or "").strip()  # puede venir vacío; JS usa STOP_CODE_DESC si no hay desc
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

                            # guardar desc si viene (para intervalos o si el Excel trae texto)
                            if key_desc and (
                                    key_code not in other_desc_by_slot[slot] or not other_desc_by_slot[slot][key_code]):
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
                        "code": str(code),  # ✅ JS usa it.code
                        "desc": str(desc_map.get(code, "")),  # ✅ si no está, JS usa STOP_CODE_DESC
                        "seconds": sec_f,
                        "hhmm": _fmt_hhmm(sec_f),
                    }
                )
            return out

        buckets = []
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

                    # ✅ esto es lo que tu JS lee
                    "other_causes": other_items,
                }
            )

        return buckets

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
    par_f["downtime_s"] = pd.to_timedelta(par_f.get("Duracion", pd.Series(dtype="timedelta64[ns]"))).dt.total_seconds()
    par_f["downtime_s"] = pd.to_numeric(par_f["downtime_s"], errors="coerce").fillna(0.0).astype(float)

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

    # -------------------------
    # Horas activas (slots)
    # -------------------------
    active_slots: set[tuple] = set()
    hours_active = 0
    if "Fecha" in res_f.columns and "Hora" in res_f.columns and not res_f.empty:
        tmp = res_f.dropna(subset=["Fecha", "Hora"]).copy()
        tmp["Fecha_d"] = pd.to_datetime(tmp["Fecha"], errors="coerce").dt.date
        tmp["Hora_h"] = tmp["Hora"].apply(lambda t: int(t.hour) if hasattr(t, "hour") else np.nan)
        tmp = tmp.dropna(subset=["Fecha_d", "Hora_h"])
        active_slots = set(zip(tmp["Fecha_d"].tolist(), tmp["Hora_h"].astype(int).tolist()))
        hours_active = int(len(active_slots))

    # -------------------------
    # Tiempos
    # -------------------------
    dead_total_s = float(par_f["downtime_s"].sum()) if "downtime_s" in par_f.columns else 0.0
    productive_simple_s = float(hours_active) * 3600.0 - dead_total_s
    productive_s = max(0.0, productive_simple_s)

    par_f["CausaCode"] = pd.to_numeric(par_f.get("Causa", np.nan), errors="coerce")
    planned_s = float(par_f.loc[(par_f["CausaCode"] >= 100) & (par_f["CausaCode"] <= 199), "downtime_s"].sum())
    unplanned_s = float(par_f.loc[(par_f["CausaCode"] >= 200) & (par_f["CausaCode"] <= 299), "downtime_s"].sum())

    # -------------------------
    # Pareto
    # -------------------------
    pareto = build_pareto_paradas(par_f, top_n=10)

    # =========================================================
    # ✅ NUEVO: “Causas / Motivos” para el click en “Otro”
    # (reusamos pareto.items para no duplicar lógica)
    # =========================================================
    other_causes = pareto.get("items", []) if isinstance(pareto, dict) else []
    if other_causes is None:
        other_causes = []

    # -------------------------
    # Terminales más usadas
    # -------------------------
    terminal_usage = _build_terminal_usage(res_f, top_n=0)

    # -------------------------
    # Productividad por hora (solo DAY)
    # -------------------------
    hourly_buckets: list[dict] = []
    if (period or "").strip().lower() == "day":
        hourly_buckets = _build_hourly_buckets(res_f, par_f)

    # KPIs UI
    kpis_ui: Dict[str, str] = {
        "Crimpados (Pulsos)": _fmt_int_es(pulses_total),
        "Pulsos Carrete": f"{_fmt_int_es(carrete)} ({_fmt_pct_es(pct_carrete)})",
        "Pulsos Manual": f"{_fmt_int_es(manual)} ({_fmt_pct_es(pct_manual)})",
        "Horas activas": _fmt_int_es(hours_active),
        "Otro Tiempo de Ciclo (Muerto)": _fmt_hhmm(dead_total_s),
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

        # ✅ NUEVO (compat UI): lista directa para “Otro”
        "other_causes": other_causes,

        "kpis": {
            "hourly_buckets": hourly_buckets,

            # ✅ NUEVO: el frontend puede leerlo desde kpis.*
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

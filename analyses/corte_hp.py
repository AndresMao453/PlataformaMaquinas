# analyses/corte_hp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import re
import numpy as np

ANALYSIS_KEY = "corte_hp"
ANALYSIS_TITLE = "Análisis Línea de Corte (HP)"

def _sec_to_ui_hours_2(sec: int) -> float:
    """Convierte segundos a horas en decimal redondeado a 2 decimales (como el UI)."""
    try:
        s = float(sec or 0)
    except Exception:
        s = 0.0
    if s < 0:
        s = 0.0
    return round(s / 3600.0, 2)


def _ui_hours_2_to_sec(h: float) -> int:
    """Convierte horas en decimal (2 decimales) a segundos."""
    try:
        x = float(h or 0.0)
    except Exception:
        x = 0.0
    if x < 0:
        x = 0.0
    return int(round(x * 3600.0))


def _balance_no_reg_sec_ui(hd_sec: int, prod_sec: int, other_sec: int) -> int:
    """
    Balancea No registrado para que en el UI (2 decimales) se cumpla:
      HD == Efectivo + Otro + No registrado
    """
    hd_h = _sec_to_ui_hours_2(hd_sec)
    prod_h = _sec_to_ui_hours_2(prod_sec)
    other_h = _sec_to_ui_hours_2(other_sec)

    no_reg_h = round(hd_h - (prod_h + other_h), 2)
    if no_reg_h < 0:
        no_reg_h = 0.0

    return _ui_hours_2_to_sec(no_reg_h)


# ============================================================
# Utilidades
# ============================================================

def _norm(s: Any) -> str:
    return str(s).strip().lower()


def _find_col(df: pd.DataFrame, contains: List[str]) -> Optional[str]:
    cols = {_norm(c): c for c in df.columns}
    for k, orig in cols.items():
        if all(tok in k for tok in contains):
            return orig
    for k, orig in cols.items():
        if any(tok in k for tok in contains):
            return orig
    return None


def _parse_hhmm_to_seconds(hhmm: str) -> int:
    s = str(hhmm or "").strip()
    if not s:
        return 0
    m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})", s)
    if not m:
        return 0
    try:
        hh = int(m.group(1))
        mm = int(m.group(2))
    except Exception:
        return 0
    hh = max(0, min(23, hh))
    mm = max(0, min(59, mm))
    return hh * 3600 + mm * 60


def _seconds_to_hhmm(sec: float) -> str:
    if sec is None or pd.isna(sec):
        return ""
    # redondeo al minuto (evita 00:59 por 3599s)
    s = int(round(max(0.0, float(sec)) / 60.0) * 60)
    hh = s // 3600
    mm = (s % 3600) // 60
    return f"{hh:02d}:{mm:02d}"



def _seconds_to_hhmmss(sec: int) -> str:
    s = int(max(0, sec or 0))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def _format_hmm_from_seconds(sec: float) -> str:
    if sec is None or pd.isna(sec) or sec < 0:
        return "0:00"
    s = int(round(float(sec) / 60.0) * 60)  # redondeo al minuto
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}:{m:02d}"



def _excel_time_to_seconds(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype="float64")
    if pd.api.types.is_numeric_dtype(s):
        td = pd.to_timedelta(s, unit="D", errors="coerce")
        return td.dt.total_seconds()
    ss = s.astype(str).str.strip()
    td = pd.to_timedelta(ss, errors="coerce")
    return td.dt.total_seconds()


def _build_dt(df: pd.DataFrame, col_fecha: Optional[str], col_hora: Optional[str]) -> pd.Series:
    if not col_fecha or col_fecha not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")

    fecha_series = df[col_fecha]
    dt_fecha = pd.to_datetime(fecha_series, errors="coerce", dayfirst=True)

    if dt_fecha.isna().any():
        txt = fecha_series.astype(str).str.strip()
        iso_mask = txt.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
        dt_iso = pd.to_datetime(txt.where(iso_mask), errors="coerce", yearfirst=True, dayfirst=False)
        dt_dmy = pd.to_datetime(txt.where(~iso_mask), errors="coerce", dayfirst=True)
        dt_fecha = dt_fecha.fillna(dt_iso).fillna(dt_dmy)

    dt_fecha = dt_fecha.dt.normalize()

    if not col_hora or col_hora not in df.columns:
        return dt_fecha

    sec = _excel_time_to_seconds(df[col_hora]).fillna(0)
    return dt_fecha + pd.to_timedelta(sec, unit="s")


def _is_general_operator(op: str) -> bool:
    op = (op or "").strip().lower()
    return (op == "") or (op == "general")


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on", "si", "sí")


# ============================================================
# Diccionario fijo: Código -> Descripción (Paradas)
# ============================================================

STOP_CODE_DESC: Dict[str, str] = {
    "100": "PARADAS PLANEADAS",
    "101": "Reuniones/Capacitación",
    "104": "Pausas activas",
    "105": "Almuerzo/Desayuno/Refrigerio horas extras",
    "106": "Baños/Pausa para tomar Agua",
    "108": "Mantenimiento Preventivo",
    "110": "Orden y Aseo General de Puesto",
    "111": "Puestas a punto/Pruebas de calidad",
    "200": "PARADAS NO PLANEADAS",
    "201": "Falta de energía/Aire comprimido",
    "203": "Cambio de chipas o carretes",
    "204": "Parada por daño en el aplicador",
    "205": "Parada por daño máquina de aplicación/uniones/manguera/corte/video jet",
    "206": "Cables enredados/Empalmar cables",
    "219": "Parada por daño en software o hardware/ PLC",
    "221": "Parada por falta de insumos o materiales",
    "222": "Problema/Falta de Información (Planillas, Desarrollos o Planos/tarjetas, tableros)",
    "224": "Parada por materiales defectuosos o fuera de especificación",
    "226": "Faltante de Circuito/ Error en el Circuito",
    "235": "Accidente/incidentes",
    "237": "Tiempo espera mantenimiento",
    "238": "Ajuste de aplicador-Adaptaciones",
    "240": "corte de terminal manual",
    "241": "Cambio de aplicador",
}


def _code_to_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = re.search(r"\d+", s)
    if not m:
        return ""
    try:
        return str(int(m.group(0)))
    except Exception:
        return m.group(0)


def _desc_from_dict(code_str: str) -> str:
    c = _code_to_str(code_str)
    return STOP_CODE_DESC.get(str(c).strip(), "")

def _clip_paradas_to_work_window_hp(
    df_par: Optional[pd.DataFrame],
    r0: pd.Timestamp,
    r1: pd.Timestamp,
    time_start: str,
    time_end: str,
) -> Optional[pd.DataFrame]:
    """
    Recorta paradas a la ventana trabajada EXACTA por día:
      - Primero recorta a rango [r0, r1]
      - Luego, por cada día, recorta a [cap_start, cap_end) según time_start/time_end
    Esto garantiza que Pareto/KPIs/Tarjetas vean los mismos tiempos.
    """
    if df_par is None or df_par.empty:
        return df_par

    p = df_par.copy()
    if "dt_start" not in p.columns or "dt_end" not in p.columns:
        return p

    p["dt_start"] = pd.to_datetime(p["dt_start"], errors="coerce")
    p["dt_end"] = pd.to_datetime(p["dt_end"], errors="coerce")
    p = p.dropna(subset=["dt_start", "dt_end"]).copy()
    p = p[p["dt_end"] > p["dt_start"]].copy()
    if p.empty:
        return p

    r0 = pd.Timestamp(r0)
    r1 = pd.Timestamp(r1)
    # rango global (inclusivo)
    p.loc[p["dt_start"] < r0, "dt_start"] = r0
    p.loc[p["dt_end"] > r1, "dt_end"] = r1
    p = p[p["dt_end"] > p["dt_start"]].copy()
    if p.empty:
        return p

    days = pd.date_range(r0.normalize(), r1.normalize(), freq="D")
    out_parts = []

    for day in days:
        cap_bounds = _cap_bounds_for_day(day, time_start, time_end)
        if not cap_bounds:
            continue
        cap_start, cap_end = cap_bounds  # cap_end EXCLUSIVO

        # intersecta con ese día
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta(days=1)

        m = (p["dt_end"] > day_start) & (p["dt_start"] < day_end)
        if not m.any():
            continue

        pd_ = p.loc[m].copy()

        # clip al día
        pd_.loc[pd_["dt_start"] < day_start, "dt_start"] = day_start
        pd_.loc[pd_["dt_end"] > day_end, "dt_end"] = day_end

        # clip a ventana trabajada
        pd_.loc[pd_["dt_start"] < cap_start, "dt_start"] = cap_start
        pd_.loc[pd_["dt_end"] > cap_end, "dt_end"] = cap_end

        pd_ = pd_[pd_["dt_end"] > pd_["dt_start"]].copy()
        if not pd_.empty:
            out_parts.append(pd_)

    if not out_parts:
        return p.iloc[0:0].copy()

    out = pd.concat(out_parts, ignore_index=True)

    # recalcula dur_sec si existe (para Pareto/otras)
    if "dur_sec" in out.columns:
        out["dur_sec"] = (out["dt_end"] - out["dt_start"]).dt.total_seconds()
        out["dur_sec"] = pd.to_numeric(out["dur_sec"], errors="coerce").fillna(0.0).clip(lower=0.0)

    return out


def _apply_time_window(dt: pd.Series, time_start: str, time_end: str) -> pd.Series:
    """
    Filtra dt por ventana [time_start, time_end] pero:
    - time_end incluye TODO el minuto (HH:MM:59) para no perder registros con segundos.
    - soporta cruce de medianoche (si end < start).
    """
    if dt is None or len(dt) == 0:
        return pd.Series([], dtype=bool)

    d = pd.to_datetime(dt, errors="coerce")
    valid = d.notna()

    ts = (time_start or "").strip()
    te = (time_end or "").strip()

    if (not ts) and (not te):
        out = pd.Series([True] * len(d), index=getattr(dt, "index", None))
        return out

    s0 = _parse_hhmm_to_seconds(ts)
    s1 = _parse_hhmm_to_seconds(te)

    s1_end = s1 + 59
    if s1_end > 86399:
        s1_end = 86399

    sec = (d.dt.hour * 3600 + d.dt.minute * 60 + d.dt.second).astype("float64")

    if s1_end >= s0:
        mask = (sec >= s0) & (sec <= s1_end)
    else:
        mask = (sec >= s0) | (sec <= s1_end)

    out = (mask & valid).fillna(False)
    out.index = getattr(dt, "index", out.index)
    return out


def _hhmm(t: str) -> str:
    s = (t or "").strip()
    if not s:
        return ""
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if not m:
        return ""
    return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"


def _should_apply_time_filter(time_start: str, time_end: str, dt_src: pd.Series) -> bool:
    """
    Solo aplicar filtro de horas si el usuario CAMBIÓ la ventana.
    Si time_start/time_end == (dt_min/dt_max) del dataset que se está mostrando,
    entonces es “default UI” y NO debe filtrar.
    """
    ts = _hhmm(time_start)
    te = _hhmm(time_end)
    if not ts or not te:
        return False
    if dt_src is None or dt_src.dropna().empty:
        return True

    dmin = pd.to_datetime(dt_src, errors="coerce").min()
    dmax = pd.to_datetime(dt_src, errors="coerce").max()
    if pd.isna(dmin) or pd.isna(dmax):
        return True

    default_start = dmin.strftime("%H:%M")
    default_end = dmax.strftime("%H:%M")

    if ts == default_start and te == default_end:
        return False

    return True


# ============================================================
# Duración -> segundos (ROBUSTO)
# ============================================================

def _duration_to_seconds(dur: pd.Series) -> pd.Series:
    """
    Convierte la columna Duración a segundos de forma robusta.
    """
    if dur is None:
        return pd.Series([], dtype="float64")

    if pd.api.types.is_timedelta64_dtype(dur):
        sec = dur.dt.total_seconds().astype("float64").fillna(0.0)
        sec[sec < 0] = 0.0
        return sec

    if pd.api.types.is_numeric_dtype(dur):
        x = pd.to_numeric(dur, errors="coerce").astype("float64")
        out = pd.Series(np.nan, index=x.index, dtype="float64")

        mexcel = (x > 0) & (x <= 2.0)
        out.loc[mexcel] = x.loc[mexcel] * 86400.0

        rest = ~mexcel & x.notna()

        mus = rest & (x >= 1e8)
        out.loc[mus] = x.loc[mus] / 1e6

        rem = rest & ~mus

        ms = rem & (x >= 1e6)
        out.loc[ms] = x.loc[ms] / 1e3

        out.loc[rem & ~ms] = x.loc[rem & ~ms]

        still = out.notna() & (out > 2 * 86400)
        out.loc[still] = x.loc[still] / 1e6

        out = out.fillna(0.0)
        out[out < 0] = 0.0
        return out.astype("float64")

    s = dur.astype(str).str.strip()
    td = pd.to_timedelta(s, errors="coerce")
    sec = td.dt.total_seconds()

    m = sec.isna()
    if m.any():
        xs = pd.to_numeric(s.where(m), errors="coerce")
        if xs.notna().any():
            sec = sec.fillna(_duration_to_seconds(xs))

    sec = sec.fillna(0.0).astype("float64")
    sec[sec < 0] = 0.0
    return sec


# ============================================================
# KPI "Otro Tiempo de Ciclo (Muerto)" desde hoja Duración
# ============================================================

def _coerce_excel_date_series_best(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype="datetime64[ns]")

    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce").dt.normalize()

    if pd.api.types.is_numeric_dtype(s):
        dt = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
        return pd.to_datetime(dt, errors="coerce").dt.normalize()

    dt0 = pd.to_datetime(s, errors="coerce")
    if dt0.isna().any():
        txt = s.astype(str).str.strip()
        dt_dmy = pd.to_datetime(txt, errors="coerce", dayfirst=True)
        dt0 = dt0.fillna(dt_dmy)

        num = pd.to_numeric(txt, errors="coerce")
        mnum = num.notna() & dt0.isna()
        if mnum.any():
            dt0.loc[mnum] = pd.to_datetime(num.loc[mnum], unit="D", origin="1899-12-30", errors="coerce")

    return pd.to_datetime(dt0, errors="coerce").dt.normalize()


def _sum_dead_seconds_from_duracion_sheet(
    df_duracion: Optional[pd.DataFrame],
    r0: pd.Timestamp,
    r1: pd.Timestamp,
) -> int:
    if df_duracion is None or not isinstance(df_duracion, pd.DataFrame) or df_duracion.empty:
        return 0
    if df_duracion.shape[1] < 4:
        return 0

    data = df_duracion.iloc[2:].copy()
    if data.empty:
        return 0

    dur = pd.to_numeric(data.iloc[:, 3], errors="coerce").fillna(0.0)
    dur = dur.where(dur > 0, 0.0)

    best_dt = None
    best_ratio = 0.0

    for j in (0, 1, 2):
        if j >= data.shape[1]:
            continue
        dt = _coerce_excel_date_series_best(data.iloc[:, j])
        ratio = float(dt.notna().mean()) if len(dt) else 0.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_dt = dt

    r0d = pd.Timestamp(r0).normalize()
    r1d = pd.Timestamp(r1).normalize()

    if best_dt is not None and best_ratio >= 0.20:
        m = best_dt.notna() & (best_dt >= r0d) & (best_dt <= r1d)
        total = float(dur.loc[m].sum()) if m.any() else 0.0
        return int(round(total))

    return int(round(float(dur.sum())))


# ============================================================
# Columnas HP
# ============================================================

@dataclass
class HPVerifCols:
    col_fecha: Optional[str]
    col_hora: Optional[str]
    col_nombre: Optional[str]


@dataclass
class HPCorteCols:
    col_fecha: Optional[str]
    col_hora: Optional[str]
    col_total: Optional[str]
    col_solic: Optional[str]
    col_buenas: Optional[str]


@dataclass
class HPParadasCols:
    col_fecha: Optional[str]
    col_hora: Optional[str]
    col_codigo: Optional[str]
    col_duracion: Optional[str]


def _resolve_hp_verif_cols(df: pd.DataFrame) -> HPVerifCols:
    return HPVerifCols(
        col_fecha=_find_col(df, ["fecha"]),
        col_hora=_find_col(df, ["hora"]),
        col_nombre=_find_col(df, ["nombre"]),
    )


def _resolve_hp_corte_cols(df: pd.DataFrame) -> HPCorteCols:
    return HPCorteCols(
        col_fecha=_find_col(df, ["fecha"]),
        col_hora=_find_col(df, ["hora"]),
        col_total=_find_col(df, ["total", "piezas"]) or _find_col(df, ["total"]),
        col_solic=_find_col(df, ["solicitadas"]) or _find_col(df, ["solicitadas", "usuario"]) or _find_col(df, ["piezas", "solicitadas"]),
        col_buenas=_find_col(df, ["piezas", "buenas"]) or _find_col(df, ["buenas"]),
    )


def _resolve_hp_paradas_cols(df: pd.DataFrame) -> HPParadasCols:
    # ✅ CAMBIO MÍNIMO: detectar mejor "código" en headers raros (paro/causa/motivo)
    return HPParadasCols(
        col_fecha=_find_col(df, ["fecha"]),
        col_hora=_find_col(df, ["hora"]),
        col_codigo=(
            _find_col(df, ["código"]) or _find_col(df, ["codigo"]) or
            _find_col(df, ["cod"]) or
            _find_col(df, ["paro"]) or _find_col(df, ["parada"]) or
            _find_col(df, ["causa"]) or _find_col(df, ["motivo"])
        ),
        col_duracion=_find_col(df, ["duración"]) or _find_col(df, ["duracion"]) or _find_col(df, ["dur"]) or _find_col(df, ["tiempo"]),
    )


# ============================================================
# Asignación de operaria por timeline (merge_asof)
# ============================================================

def _build_operator_timeline(df_verif: pd.DataFrame) -> Tuple[pd.DataFrame, HPVerifCols, pd.Series]:
    cols = _resolve_hp_verif_cols(df_verif)
    dt = _build_dt(df_verif, cols.col_fecha, cols.col_hora)

    if not cols.col_nombre or cols.col_nombre not in df_verif.columns:
        tl = pd.DataFrame(columns=["dt", "op"])
        return tl, cols, dt

    tl = pd.DataFrame({
        "dt": dt,
        "op": df_verif[cols.col_nombre].astype(str).str.strip(),
    }).dropna(subset=["dt"])

    tl = tl[tl["op"].astype(str).str.strip().ne("")].copy()
    tl = tl[~tl["op"].str.lower().isin(["nan", "none"])].copy()
    tl = tl.sort_values("dt")

    return tl, cols, dt


def _assign_operator_by_asof(
    df: pd.DataFrame,
    dt: pd.Series,
    timeline: pd.DataFrame,
    *,
    dt_colname: str = "dt",
) -> pd.Series:
    if timeline is None or timeline.empty or dt is None or dt.dropna().empty:
        return pd.Series([""] * len(df), index=df.index, dtype="object")

    d = pd.to_datetime(dt, errors="coerce")

    base = pd.DataFrame({dt_colname: d}).copy()
    base["idx"] = df.index.to_numpy()
    base = base.dropna(subset=[dt_colname]).sort_values(dt_colname)
    if base.empty:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    base["_day"] = base[dt_colname].dt.normalize()

    tl = timeline[["dt", "op"]].copy()
    tl["dt"] = pd.to_datetime(tl["dt"], errors="coerce")
    tl["op"] = tl["op"].fillna("").astype(str).str.strip()
    tl = tl.dropna(subset=["dt"]).sort_values("dt")
    if tl.empty:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    tl["_day"] = tl["dt"].dt.normalize()

    merged_back = pd.merge_asof(
        base,
        tl,
        left_on=dt_colname,
        right_on="dt",
        by="_day",
        direction="backward",
        allow_exact_matches=True,
    )

    out = pd.Series([""] * len(df), index=df.index, dtype="object")
    if not merged_back.empty:
        idxs = merged_back["idx"].to_numpy()
        ops = merged_back["op"].fillna("").astype(str).str.strip().to_numpy()
        out.loc[idxs] = ops

    m_blank = out.astype(str).str.strip().eq("")
    if m_blank.any():
        base_blank = base[base["idx"].isin(out.index[m_blank])].copy()
        if not base_blank.empty:
            merged_fwd = pd.merge_asof(
                base_blank,
                tl,
                left_on=dt_colname,
                right_on="dt",
                by="_day",
                direction="forward",
                allow_exact_matches=True,
            )
            if not merged_fwd.empty:
                idxs2 = merged_fwd["idx"].to_numpy()
                ops2 = merged_fwd["op"].fillna("").astype(str).str.strip().to_numpy()
                out.loc[idxs2] = ops2

    return out


# ============================================================
# Paradas: header + escoger columna Duración + intervalos
# ============================================================

def _ensure_header_row(df: pd.DataFrame, max_scan: int = 30) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    cols_norm = [str(c).strip().lower() for c in df.columns]
    if any("fecha" in c for c in cols_norm) and any("hora" in c for c in cols_norm):
        return df

    for i in range(min(max_scan, len(df))):
        row = df.iloc[i].astype(str).str.strip().str.lower().tolist()
        if ("fecha" in row) and ("hora" in row):
            new_cols = [str(x).strip() for x in df.iloc[i].tolist()]
            out = df.iloc[i + 1:].copy()
            out.columns = new_cols
            return out

    return df


def _pick_best_duration_col(df: pd.DataFrame, cols: HPParadasCols) -> Optional[str]:
    if df is None or df.empty:
        return None

    fecha = cols.col_fecha
    hora = cols.col_hora
    codigo = cols.col_codigo

    cand = []
    for c in df.columns:
        k = _norm(c)
        if c in (fecha, hora, codigo):
            continue
        if any(t in k for t in ["dur", "duración", "duracion", "tiempo", "min", "seg"]):
            cand.append(c)

    if cols.col_duracion and cols.col_duracion in df.columns:
        if cols.col_duracion not in cand:
            cand.insert(0, cols.col_duracion)

    if not cand and len(df.columns) >= 4:
        cand = [df.columns[3]]

    best = None
    best_score = -1e18

    for c in cand:
        try:
            sec = _duration_to_seconds(df[c])
        except Exception:
            continue

        sec = pd.to_numeric(sec, errors="coerce").fillna(0.0)
        nz = int((sec > 0).sum())
        if nz == 0:
            continue

        med = float(sec[sec > 0].median()) if (sec > 0).any() else 0.0
        p95 = float(sec.quantile(0.95)) if len(sec) else 0.0
        mx = float(sec.max()) if len(sec) else 0.0

        score = nz * 10.0
        if 5 <= med <= 7200:
            score += 5000
        if p95 > 6 * 3600:
            score -= 8000
        if mx > 24 * 3600:
            score -= 12000

        if score > best_score:
            best_score = score
            best = c

    return best


def _build_paradas_intervals(df_paradas: pd.DataFrame) -> pd.DataFrame:
    df_paradas = _ensure_header_row(df_paradas, max_scan=30)
    cols = _resolve_hp_paradas_cols(df_paradas)

    if not cols.col_fecha or not cols.col_hora:
        return pd.DataFrame(columns=["dt_start", "dt_end", "codigo", "dur_sec"])

    dur_col = _pick_best_duration_col(df_paradas, cols)
    if not dur_col:
        return pd.DataFrame(columns=["dt_start", "dt_end", "codigo", "dur_sec"])

    dt_end = _build_dt(df_paradas, cols.col_fecha, cols.col_hora)

    dur_sec = _duration_to_seconds(df_paradas[dur_col]).astype("float64")
    dt_start = dt_end - pd.to_timedelta(dur_sec, unit="s")

    if cols.col_codigo and cols.col_codigo in df_paradas.columns:
        codigo = df_paradas[cols.col_codigo].apply(_code_to_str)
    else:
        codigo = pd.Series([""] * len(df_paradas), index=df_paradas.index)

    out = pd.DataFrame(
        {
            "dt_start": dt_start,
            "dt_end": dt_end,
            "codigo": codigo.astype(str).str.strip(),
            "dur_sec": dur_sec,
        }
    )

    out = out.dropna(subset=["dt_end", "dt_start"])
    out["dur_sec"] = pd.to_numeric(out["dur_sec"], errors="coerce").fillna(0.0).astype(float)
    out = out[(out["dur_sec"] > 0) & (out["dt_end"] > out["dt_start"])].copy()
    out = out[out["codigo"].astype(str).str.strip().ne("")].copy()

    return out


def _sum_paradas_by_dtend_seconds_simple(
    df_paradas_iv: Optional[pd.DataFrame],
    r0: pd.Timestamp,
    r1: pd.Timestamp,
) -> Tuple[int, int, int]:
    MEAL_CODE = "105"

    if df_paradas_iv is None or df_paradas_iv.empty:
        return 0, 0, 0

    p = df_paradas_iv.dropna(subset=["dt_end"]).copy()
    if p.empty:
        return 0, 0, 0

    r0 = pd.Timestamp(r0)
    r1 = pd.Timestamp(r1)

    p = p[(p["dt_end"] >= r0) & (p["dt_end"] <= r1)].copy()
    if p.empty:
        return 0, 0, 0

    dur = pd.to_numeric(p.get("dur_sec", 0), errors="coerce").fillna(0.0)
    dur[dur < 0] = 0.0

    total = float(dur.sum())

    codes = p.get("codigo", "").astype(str).str.strip()
    meal = float(dur.loc[codes == MEAL_CODE].sum()) if len(codes) == len(dur) else 0.0

    other = max(0.0, total - meal)

    return int(round(other)), int(round(meal)), int(round(total))


# ============================================================
# Periodo -> rango (day/week/month)
# ============================================================

def _range_from_period(
    period: str,
    period_value: str,
    fallback_dt: Optional[pd.Timestamp],
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    period = (period or "day").strip().lower()
    pv = str(period_value or "").strip().replace("/", "-")

    def _fallback() -> pd.Timestamp:
        if fallback_dt is not None and not pd.isna(fallback_dt):
            return pd.Timestamp(fallback_dt).normalize()
        return pd.Timestamp.today().normalize()

    if period == "day":
        d = pd.to_datetime(pv, errors="coerce")
        if pd.isna(d):
            d = _fallback()
        d = pd.Timestamp(d).normalize()
        return d, d + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

    if period == "week":
        pv_l = pv.lower()
        if re.match(r"^\d{4}-w\d{1,2}$", pv_l):
            y, w = pv_l.split("-w")
            y = int(y)
            w = int(w)
            start = pd.Timestamp.fromisocalendar(y, w, 1)
        else:
            d = pd.to_datetime(pv, errors="coerce")
            if pd.isna(d):
                d = _fallback()
            d = pd.Timestamp(d).normalize()
            start = d - pd.Timedelta(days=int(d.weekday()))
        end = start + pd.Timedelta(days=7) - pd.Timedelta(milliseconds=1)
        return start, end

    if period == "month":
        if re.match(r"^\d{4}-\d{2}$", pv):
            start = pd.Timestamp(f"{pv}-01")
        else:
            d = pd.to_datetime(pv, errors="coerce")
            if pd.isna(d):
                d = _fallback()
            d = pd.Timestamp(d).normalize()
            start = pd.Timestamp(f"{d.year:04d}-{d.month:02d}-01")
        end = (start + pd.offsets.MonthBegin(1)) - pd.Timedelta(milliseconds=1)
        return start, end

    d = pd.to_datetime(pv, errors="coerce")
    if pd.isna(d):
        d = _fallback()
    d = pd.Timestamp(d).normalize()
    return d, d + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)


# ============================================================
# Capacidad (horas registradas)
# ============================================================
def _cap_bounds_for_day(
    day: pd.Timestamp,
    time_start: str,
    time_end: str
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Devuelve bounds de capacidad como intervalo exacto [cap_start, cap_end) (end EXCLUSIVO).
    Regla: time_end incluye TODO el minuto (HH:MM:00..HH:MM:59) => cap_end = end + 60s.
    """
    day = pd.Timestamp(day).normalize()
    if not time_start or not time_end:
        return None

    s0 = _parse_hhmm_to_seconds(time_start)
    s1 = _parse_hhmm_to_seconds(time_end)

    # si llega raro (end <= start), al menos 1 minuto
    if s1 <= s0:
        s1 = s0

    cap_start = day + pd.to_timedelta(s0, unit="s")

    # end incluye el minuto completo => end_exclusive = end + 60s
    end_excl_sec = s1 + 60
    cap_end = day + pd.to_timedelta(end_excl_sec, unit="s")

    # cap_end no puede pasar de +1 día
    day_end = day + pd.Timedelta(days=1)
    if cap_end > day_end:
        cap_end = day_end

    if cap_end <= cap_start:
        return None

    return cap_start, cap_end



def _capacity_seconds_for_range_hp(
    r0: pd.Timestamp,
    r1: pd.Timestamp,
    time_start: str,
    time_end: str
) -> int:
    if not time_start or not time_end:
        return 0

    r0 = pd.Timestamp(r0)
    r1 = pd.Timestamp(r1)
    if pd.isna(r0) or pd.isna(r1) or r1 < r0:
        return 0

    # convertir rango inclusivo a exclusivo (si viene con -1ms típico)
    r1_excl = r1 + pd.Timedelta(milliseconds=1)

    s0 = _parse_hhmm_to_seconds(time_start)
    s1 = _parse_hhmm_to_seconds(time_end)
    if s1 <= s0:
        s1 = s0

    # end incluye minuto completo => end_exclusive = end + 60s
    end_excl_sec = s1 + 60
    if end_excl_sec > 24 * 3600:
        end_excl_sec = 24 * 3600

    d0 = r0.normalize()
    d1 = r1.normalize()

    total = 0
    day = d0
    one = pd.Timedelta(days=1)

    while day <= d1:
        cap_start = day + pd.to_timedelta(s0, unit="s")
        cap_end = day + pd.to_timedelta(end_excl_sec, unit="s")  # EXCLUSIVO

        # recorte al rango [r0, r1_excl)
        a0 = max(cap_start, r0)
        a1 = min(cap_end, r1_excl)
        if a1 > a0:
            total += int((a1.value - a0.value) // 1_000_000_000)

        day = day + one

    return int(total)


def _derive_time_bounds_from_activity(dt_activity: pd.Series, r0: pd.Timestamp, r1: pd.Timestamp) -> Tuple[str, str]:
    if dt_activity is None or dt_activity.dropna().empty:
        return "", ""
    d = pd.to_datetime(dt_activity, errors="coerce").dropna()
    d = d[(d >= r0) & (d <= r1)]
    if d.empty:
        return "", ""
    tod = (d - d.dt.normalize()).dt.total_seconds()
    mn = int(tod.min())
    mx = int(tod.max())
    return _seconds_to_hhmm(float(mn)), _seconds_to_hhmm(float(mx))


# ============================================================
# Pareto Paradas (por código)
# ============================================================

def _pareto_paradas_hp(df_paradas_iv: Optional[pd.DataFrame], top_n: int = 25) -> Dict[str, Any]:
    if df_paradas_iv is None or df_paradas_iv.empty:
        return {"labels": [], "descs": [], "dur_sec": [], "dur_hhmm": [], "cum_pct": [], "total_sec": 0}

    if "codigo" not in df_paradas_iv.columns:
        return {"labels": [], "descs": [], "dur_sec": [], "dur_hhmm": [], "cum_pct": [], "total_sec": 0}

    d = df_paradas_iv.copy()
    d["codigo"] = d["codigo"].apply(_code_to_str)
    if "dur_sec" in d.columns:
        d["dur_sec"] = pd.to_numeric(d["dur_sec"], errors="coerce").fillna(0.0).astype(float)
    else:
        ds = pd.to_datetime(d.get("dt_start"), errors="coerce")
        de = pd.to_datetime(d.get("dt_end"), errors="coerce")
        d["dur_sec"] = (de - ds).dt.total_seconds()
        d["dur_sec"] = pd.to_numeric(d["dur_sec"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)

    d = d[(d["codigo"].astype(str).str.strip().ne("")) & (d["dur_sec"] > 0)].copy()
    if d.empty:
        return {"labels": [], "descs": [], "dur_sec": [], "dur_hhmm": [], "cum_pct": [], "total_sec": 0}

    total_sec = float(d["dur_sec"].sum())

    g = (
        d.groupby("codigo", dropna=False)["dur_sec"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    if g.empty or total_sec <= 0:
        return {"labels": [], "descs": [], "dur_sec": [], "dur_hhmm": [], "cum_pct": [], "total_sec": int(round(total_sec))}

    cum_pct = (g.cumsum() / total_sec * 100.0).round(1).tolist()

    labels = [str(x) for x in g.index.tolist()]
    dur_sec = [float(x) for x in g.tolist()]
    dur_hhmm = [_seconds_to_hhmm(float(x)) for x in dur_sec]
    descs = [(_desc_from_dict(c) or "") for c in labels]

    return {
        "labels": labels,
        "descs": descs,
        "dur_sec": dur_sec,
        "dur_hhmm": dur_hhmm,
        "cum_pct": cum_pct,
        "total_sec": int(round(total_sec)),
    }


# ============================================================
# week/month = SUMA por día usando la MISMA lógica diaria (buckets)
# ============================================================

def _sum_cap_and_effective_by_day_hp(
    *,
    df_corte_src: pd.DataFrame,
    corte_cols: HPCorteCols,
    df_paradas_src: Optional[pd.DataFrame],
    r0: pd.Timestamp,
    r1: pd.Timestamp,
    time_start: str,
    time_end: str,
    apply_time_filter: bool,
) -> Tuple[int, int]:
    if df_corte_src is None or df_corte_src.empty:
        return 0, 0

    r0t = pd.Timestamp(r0)
    r1t = pd.Timestamp(r1)

    cap_total = 0
    prod_total = 0

    day = r0t.normalize()
    end_day = r1t.normalize()

    while day <= end_day:
        day_start = day
        day_end = day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

        m_day_c = df_corte_src["_dt"].notna() & (df_corte_src["_dt"] >= day_start) & (df_corte_src["_dt"] <= day_end)
        df_c_day = df_corte_src.loc[m_day_c].copy()

        df_p_day = None
        if df_paradas_src is not None and not df_paradas_src.empty:
            p = df_paradas_src.dropna(subset=["dt_start", "dt_end"]).copy()
            if not p.empty:
                m_day_p = (p["dt_end"] >= day_start) & (p["dt_start"] <= day_end)
                df_p_day = p.loc[m_day_p].copy()

        ts = time_start
        te = time_end
        if not apply_time_filter:
            activity = []
            if not df_c_day.empty:
                activity.append(df_c_day["_dt"])
            if df_p_day is not None and not df_p_day.empty:
                activity.append(df_p_day["dt_end"])
            if activity:
                dt_activity = pd.concat(activity, ignore_index=True)
                ts2, te2 = _derive_time_bounds_from_activity(dt_activity, day_start, day_end)
                if ts2 and te2:
                    ts, te = ts2, te2

        if not ts or not te:
            ts = ts or "00:00"
            te = te or "23:59"

        buckets = _compute_hourly_buckets_hp(
            df_corte=df_c_day,
            dt_corte=df_c_day["_dt"] if "_dt" in df_c_day.columns else pd.Series([], dtype="datetime64[ns]"),
            corte_cols=corte_cols,
            selected_day=day_start,
            df_paradas_iv=df_p_day,
            time_start=ts,
            time_end=te,
        )

        if buckets:
            cap_total += int(sum(b.get("capacitySec", 0) for b in buckets))
            prod_total += int(sum(b.get("prodSec", 0) for b in buckets))

        day = day + pd.Timedelta(days=1)

    return int(cap_total), int(prod_total)


# ============================================================
# Buckets por hora (DAY)
# ============================================================

def _compute_hourly_buckets_hp(
    df_corte: pd.DataFrame,
    dt_corte: pd.Series,
    corte_cols: HPCorteCols,
    selected_day: pd.Timestamp,
    df_paradas_iv: Optional[pd.DataFrame],
    *,
    time_start: str,
    time_end: str,
) -> List[Dict[str, Any]]:
    MEAL_CODE = "105"
    PAUSE_CODE = "104"   # ✅ antes era 106, ahora es 104
    NS = 1_000_000_000  # ns -> sec

    if dt_corte is None or dt_corte.empty:
        return []

    day = pd.Timestamp(selected_day).normalize()

    # ventana "trabajada" (se usa para calcular efectivo)
    cap_bounds = _cap_bounds_for_day(day, time_start, time_end)
    if not cap_bounds:
        return []
    cap_start, cap_end = cap_bounds

    day_start = day
    day_end = day + pd.Timedelta(days=1)

    # recorta ventana al día
    cap_start = max(pd.Timestamp(cap_start), day_start)
    cap_end = min(pd.Timestamp(cap_end), day_end)
    if cap_end <= cap_start:
        return []

    cap_start_ns = int(pd.Timestamp(cap_start).value)
    cap_end_ns = int(pd.Timestamp(cap_end).value)

    # ======================
    # Cortes por hora (solo por día)
    # ======================
    cut_by_hour: Dict[int, float] = {}
    c_buenas = corte_cols.col_buenas
    if c_buenas and c_buenas in df_corte.columns:
        tmp = pd.DataFrame(
            {"dt": dt_corte, "v": pd.to_numeric(df_corte[c_buenas], errors="coerce").fillna(0)}
        )
        tmp = tmp.dropna(subset=["dt"])
        tmp = tmp[(tmp["dt"] >= day_start) & (tmp["dt"] < day_end)]
        if not tmp.empty:
            cut_by_hour = tmp.groupby(tmp["dt"].dt.hour)["v"].sum().to_dict()

    # ======================
    # Paradas (arrays ns) — CLIP a la ventana trabajada
    # ======================
    stops_start_ns = None
    stops_end_ns = None
    codes_np = None

    # para detalle de causas por código (incluye 104, excluye 105)
    code_groups: Dict[str, np.ndarray] = {}

    if df_paradas_iv is not None and not df_paradas_iv.empty:
        p = df_paradas_iv.copy()
        if "dt_start" in p.columns and "dt_end" in p.columns:
            p = p.dropna(subset=["dt_start", "dt_end"]).copy()
            if not p.empty:
                p["dt_start"] = pd.to_datetime(p["dt_start"], errors="coerce")
                p["dt_end"] = pd.to_datetime(p["dt_end"], errors="coerce")
                p = p.dropna(subset=["dt_start", "dt_end"]).copy()

            if not p.empty:
                # clip al día
                p.loc[p["dt_start"] < day_start, "dt_start"] = day_start
                p.loc[p["dt_end"] > day_end, "dt_end"] = day_end

                # clip a la ventana trabajada
                p.loc[p["dt_start"] < cap_start, "dt_start"] = cap_start
                p.loc[p["dt_end"] > cap_end, "dt_end"] = cap_end

                p = p[p["dt_end"] > p["dt_start"]].copy()

            if not p.empty:
                codes_np = p.get("codigo", "").astype(str).str.strip().to_numpy()
                stops_start_ns = p["dt_start"].astype("datetime64[ns]").astype("int64").to_numpy()
                stops_end_ns = p["dt_end"].astype("datetime64[ns]").astype("int64").to_numpy()

                # agrupar índices por código (para TOP causas en tarjeta)
                tmp_groups: Dict[str, List[int]] = {}
                for i, c in enumerate(codes_np):
                    c = str(c).strip()
                    if (not c) or (c == MEAL_CODE):
                        continue  # ✅ 105 no va en causas de "Otro"
                    tmp_groups.setdefault(c, []).append(i)
                for k, v in tmp_groups.items():
                    code_groups[k] = np.array(v, dtype=np.int64)

    buckets: List[Dict[str, Any]] = []

    for h in range(24):
        h0 = day_start + pd.Timedelta(hours=h)
        h1 = h0 + pd.Timedelta(hours=1)

        h0_ns = int(h0.value)
        h1_ns = int(h1.value)

        # capacidad trabajada dentro de la ventana (para efectivo)
        work_ns = max(0, min(h1_ns, cap_end_ns) - max(h0_ns, cap_start_ns))
        if work_ns <= 0:
            continue

        # ✅ CAMBIO: redondeo ns->sec (evita 3599s y el 0.99)
        cap_work_sec = int(round(work_ns / NS))

        # capacidad full (60 min) para el "No registrado" por tarjeta
        cap_full_sec = 3600

        # =========
        # Paradas registradas dentro de la hora
        # =========
        dead_rec_sec = 0
        meal_sec = 0
        pause_sec = 0

        ol_ns = None
        if stops_start_ns is not None and stops_end_ns is not None and len(stops_start_ns) > 0:
            ol_ns = np.minimum(stops_end_ns, h1_ns) - np.maximum(stops_start_ns, h0_ns)
            ol_ns = np.maximum(0, ol_ns)

            # ✅ CAMBIO: redondeo ns->sec
            dead_rec_sec = int(round(float(ol_ns.sum()) / NS))

            if codes_np is not None and len(codes_np) == len(ol_ns):
                meal_sec = int(round(float(ol_ns[codes_np == MEAL_CODE].sum()) / NS))
                pause_sec = int(round(float(ol_ns[codes_np == PAUSE_CODE].sum()) / NS))

        # clamp por seguridad
        if dead_rec_sec > cap_work_sec:
            dead_rec_sec = cap_work_sec
            meal_sec = min(meal_sec, cap_work_sec)
            pause_sec = min(pause_sec, cap_work_sec)

        # =========
        # Tiempo efectivo por tarjeta:
        # prod = trabajado - (todas las paradas registradas)
        # =========
        cut = int(cut_by_hour.get(h, 0))
        prod_sec = max(0, cap_work_sec - dead_rec_sec)
        if cut <= 0:
            prod_sec = 0

        # =========
        # No registrado (000) SOLO para tarjetas:
        # lo que falta para cerrar 60 min visualmente
        # =========
        unreg_sec = max(0, cap_full_sec - (prod_sec + dead_rec_sec))

        # =========
        # ✅ KPI "Otro" (sin 105 ni 104) => solo registrado
        # =========
        other_kpi_rec_sec = max(0, int(dead_rec_sec - meal_sec - pause_sec))

        # =========
        # ✅ Tarjeta "Otro" (incluye 104, excluye 105) + 000 para cerrar 60
        # =========
        other_card_rec_sec = max(0, int(dead_rec_sec - meal_sec))
        other_show_sec = int(other_card_rec_sec + unreg_sec)

        # dead mostrado en tarjeta cierra a 60 con 000
        dead_show_sec = int(dead_rec_sec + unreg_sec)

        # =========
        # ✅ Reporte de causas (tarjeta):
        # incluir 104 (y cualquier otra), excluir 105
        # y SI hace falta tiempo para cerrar => meter "000 — Desconocido (x)"
        # =========
        other_causes: List[Dict[str, Any]] = []
        known_other_sec = 0

        if ol_ns is not None and code_groups:
            pairs: List[Tuple[str, int]] = []
            for code, idxs in code_groups.items():
                # ✅ CAMBIO: redondeo ns->sec
                sec = int(round(float(ol_ns[idxs].sum()) / NS))
                if sec > 0:
                    pairs.append((code, sec))

            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:8]

            other_causes = [
                {
                    "code": str(c),
                    "seconds": int(s),
                    "hhmm": _seconds_to_hhmm(float(s)),
                    "desc": STOP_CODE_DESC.get(str(c).strip(), ""),
                }
                for c, s in pairs
            ]
            known_other_sec = int(sum(int(x.get("seconds", 0) or 0) for x in other_causes))

        # ✅ 000 como resto para cerrar el "Otro" mostrado en la tarjeta
        unknown_sec = max(0, int(other_show_sec) - int(known_other_sec))
        if unknown_sec > 0:
            other_causes = [c for c in other_causes if str(c.get("code", "")).strip() != "000"]
            other_causes.insert(0, {
                "code": "000",
                "seconds": int(unknown_sec),
                "hhmm": _seconds_to_hhmm(float(unknown_sec)),
                "desc": "Desconocido",
            })
            other_causes = other_causes[:8]

        dead_h2 = _sec_to_ui_hours_2(dead_show_sec)
        prod_h2 = _sec_to_ui_hours_2(prod_sec)
        unreg_h2 = _sec_to_ui_hours_2(unreg_sec)
        other_h2 = _sec_to_ui_hours_2(other_show_sec)
        cap_h2 = _sec_to_ui_hours_2(cap_full_sec)  # SIEMPRE 1.00


        buckets.append({
            "hourLabel": f"{h:02d}:00",
            "hour": f"{h:02d}",

            # para KPI (trabajado real)
            "capacitySec": int(cap_work_sec),

            # para tarjetas
            "deadSec": int(dead_show_sec),
            "mealSec": int(meal_sec),
            "pauseSec": int(pause_sec),  # ✅ 104 para KPI/UI

            # ✅ Tarjeta "Otro" incluye 104 + 000 (pero NO 105)
            "otherDeadSec": int(other_show_sec),

            # ✅ KPI "Otro" = dead - 105 - 104 (SIN 000)
            "otherRecordedSec": int(other_kpi_rec_sec),

            # ✅ 000 => No Registrado (balanceado en KPI aparte)
            "unregSec": int(unreg_sec),

            "prodSec": int(prod_sec),

            "cut": int(cut),
            "meters": 0.0,
            "hadActivity": True,

            # ✅ Reporte de causas en tarjetas (incluye 104 + 000 si aplica)
            "other_causes": other_causes,
            "capH2": float(cap_h2),  # 1.00
            "deadH2": float(dead_h2),
            "prodH2": float(prod_h2),
            "unregH2": float(unreg_h2),
            "otherDeadH2": float(other_h2),

        })

    return buckets









# ============================================================
# KPI: Horas Disponibles (misma regla THB por "tarjetas")
# ============================================================
def _compute_horas_disponibles_hp(
    *,
    period: str,
    r0: pd.Timestamp,
    r1: pd.Timestamp,
    df_corte_p: pd.DataFrame,
    df_par_p: Optional[pd.DataFrame],
) -> int:
    """
    Reglas (igual THB):
      - Tarjetas = # horas (buckets) entre el primer y el último registro del día
      - Lun-Jue:  >=1 => 09:08 ; >=11 => 10:08
      - Vie-Sáb:  >=1 => 06:53 ; >=9  => 07:53
      - Dom: no cuenta
      - Week/Month: suma diaria
    """
    if (df_corte_p is None or df_corte_p.empty) and (df_par_p is None or df_par_p.empty):
        return 0

    r0 = pd.Timestamp(r0)
    r1 = pd.Timestamp(r1)

    # días con datos (corte + paradas)
    days = set()

    if df_corte_p is not None and not df_corte_p.empty and "_dt" in df_corte_p.columns:
        d = pd.to_datetime(df_corte_p["_dt"], errors="coerce").dropna()
        if not d.empty:
            days |= set(d.dt.normalize().tolist())

    if df_par_p is not None and not df_par_p.empty:
        for c in ("dt_start", "dt_end"):
            if c in df_par_p.columns:
                d = pd.to_datetime(df_par_p[c], errors="coerce").dropna()
                if not d.empty:
                    days |= set(d.dt.normalize().tolist())

    if not days:
        return 0

    BASE_LJ_SEC = (9 * 3600) + (8 * 60)    # 09:08
    BASE_VS_SEC = (6 * 3600) + (53 * 60)   # 06:53

    total = 0
    for day in sorted(days):
        day = pd.Timestamp(day).normalize()
        day_start = day
        day_end = day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

        # recorte al rango real
        if day_end < r0 or day_start > r1:
            continue
        a0 = max(day_start, r0)
        a1 = min(day_end, r1)

        times = []

        # corte (dt)
        if df_corte_p is not None and not df_corte_p.empty and "_dt" in df_corte_p.columns:
            dc = pd.to_datetime(df_corte_p["_dt"], errors="coerce").dropna()
            dc = dc[(dc >= a0) & (dc <= a1)]
            if not dc.empty:
                times += dc.tolist()

        # paradas (start/end)
        if df_par_p is not None and not df_par_p.empty:
            ds = pd.to_datetime(df_par_p.get("dt_start"), errors="coerce")
            de = pd.to_datetime(df_par_p.get("dt_end"), errors="coerce")
            if ds is not None and de is not None:
                m = ds.notna() & de.notna() & (de >= a0) & (ds <= a1)
                if m.any():
                    ds2 = ds.loc[m].clip(lower=a0, upper=a1)
                    de2 = de.loc[m].clip(lower=a0, upper=a1)
                    times += ds2.tolist()
                    times += de2.tolist()

        if not times:
            continue

        activity_min = min(times)
        activity_max = max(times)

        # contar "tarjetas" como # horas con solape (igual lógica THB por rango de actividad)
        cards = 0
        for h in range(24):
            h0 = day + pd.Timedelta(hours=h)
            h1 = h0 + pd.Timedelta(hours=1)
            if min(h1, activity_max) > max(h0, activity_min):
                cards += 1

        if cards <= 0:
            continue

        wd = int(day.dayofweek)  # 0 lun ... 4 vie ... 5 sab ... 6 dom

        if wd in (0, 1, 2, 3):  # Lun-Jue
            sec = BASE_LJ_SEC + (3600 if cards >= 11 else 0)
            total += sec
        elif wd in (4, 5):      # Vie-Sáb
            sec = BASE_VS_SEC + (3600 if cards >= 9 else 0)
            total += sec
        else:
            continue  # Dom

    # truncar a minuto (como el resto del proyecto)
    total = (int(total) // 60) * 60
    return int(total)

# ============================================================
# ANALISIS PRINCIPAL
# ============================================================

def analyze_hp(
    df_corte: pd.DataFrame,
    df_verif: pd.DataFrame,
    *,
    period: str = "day",
    period_value: str = "",
    machine: str = "HP",
    operator: str = "",
    time_start: str = "",
    time_end: str = "",
    time_apply: Any = False,
    df_paradas: Optional[pd.DataFrame] = None,
    df_duracion: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:

    if df_corte is None or df_corte.empty:
        return {"status": "error", "message": "HP: La hoja 'Corte' está vacía.", "machine": machine}

    MEAL_CODE = "105"
    PAUSE_CODE = "104"   # ✅ antes era 106, ahora es 104

    def _clean_hhmm(x: str) -> str:
        s = str(x or "").strip()
        if not s:
            return ""
        m = re.search(r"(\d{1,2})\s*:\s*(\d{1,2})", s)
        if not m:
            return ""
        hh = max(0, min(23, int(m.group(1))))
        mm = max(0, min(59, int(m.group(2))))
        return f"{hh:02d}:{mm:02d}"

    time_start = _clean_hhmm(time_start)
    time_end = _clean_hhmm(time_end)

    # 1) timeline operaria desde verificación
    timeline, _, _ = _build_operator_timeline(df_verif)

    # 2) Corte: dt + operaria asignada
    corte_cols = _resolve_hp_corte_cols(df_corte)
    if not corte_cols.col_fecha:
        return {"status": "error", "message": "HP: No se encontró 'Fecha' en hoja 'Corte'.", "machine": machine}

    dt_corte = _build_dt(df_corte, corte_cols.col_fecha, corte_cols.col_hora)
    op_corte = _assign_operator_by_asof(df_corte, dt_corte, timeline, dt_colname="dt")

    df_corte2 = df_corte.copy()
    df_corte2["_dt"] = dt_corte
    df_corte2["_op"] = op_corte

    # 3) Paradas: intervalos + operaria asignada por dt_end
    df_paradas_iv = None
    if df_paradas is not None and isinstance(df_paradas, pd.DataFrame) and not df_paradas.empty:
        df_paradas_iv = _build_paradas_intervals(df_paradas)
        if df_paradas_iv is not None and not df_paradas_iv.empty:
            op_par = _assign_operator_by_asof(df_paradas_iv, df_paradas_iv["dt_end"], timeline, dt_colname="dt_end")
            df_paradas_iv = df_paradas_iv.copy()
            df_paradas_iv["_op"] = op_par

    # 4) rango por periodo
    fallback = None
    if dt_corte is not None and not dt_corte.dropna().empty:
        fallback = pd.Timestamp(dt_corte.dropna().min()).normalize()

    r0, r1 = _range_from_period(period, period_value, fallback)

    # 5) Duración sheet (NO la usamos para descomponer 104/105, solo como fallback total si NO hay paradas)
    dead_from_duracion_sec = _sum_dead_seconds_from_duracion_sheet(df_duracion, r0, r1)
    use_duracion = dead_from_duracion_sec > 0

    # 6) filtrar por periodo
    m_range_c = df_corte2["_dt"].notna() & (df_corte2["_dt"] >= r0) & (df_corte2["_dt"] <= r1)
    df_corte_p = df_corte2.loc[m_range_c].copy()

    df_par_p = None
    if df_paradas_iv is not None and not df_paradas_iv.empty:
        m_range_p = (
            df_paradas_iv["dt_end"].notna()
            & df_paradas_iv["dt_start"].notna()
            & (df_paradas_iv["dt_end"] >= r0)
            & (df_paradas_iv["dt_start"] <= r1)
        )
        df_par_p = df_paradas_iv.loc[m_range_p].copy()

    # 7) filtro por operaria + autoderivar horas si no vienen
    sel_op = str(operator or "").strip()
    if not _is_general_operator(sel_op):
        df_corte_p = df_corte_p.loc[df_corte_p["_op"].astype(str).str.strip() == sel_op].copy()
        if df_par_p is not None and not df_par_p.empty and "_op" in df_par_p.columns:
            df_par_p = df_par_p.loc[df_par_p["_op"].astype(str).str.strip() == sel_op].copy()

        if (not time_start) or (not time_end):
            activity = [df_corte_p["_dt"]]
            if df_par_p is not None and not df_par_p.empty:
                activity.append(df_par_p["dt_end"])
            dt_activity = pd.concat(activity, ignore_index=True)
            ts2, te2 = _derive_time_bounds_from_activity(dt_activity, r0, r1)
            if ts2 and te2:
                time_start, time_end = ts2, te2
    else:
        if (not time_start) or (not time_end):
            activity = [df_corte_p["_dt"]]
            if df_par_p is not None and not df_par_p.empty:
                activity.append(df_par_p["dt_end"])
            dt_activity = pd.concat(activity, ignore_index=True)
            ts2, te2 = _derive_time_bounds_from_activity(dt_activity, r0, r1)
            if ts2 and te2:
                time_start, time_end = ts2, te2

    if not time_start or not time_end:
        time_start = time_start or "00:00"
        time_end = time_end or "23:59"
        # ✅ Paradas recortadas EXACTO a la ventana trabajada (para que KPIs/Pareto/Tarjetas cuadren)

    df_par_work = _clip_paradas_to_work_window_hp(df_par_p, r0, r1, time_start, time_end)

    # 8) aplicar ventana horaria SOLO si el front dice time_apply=True
    dt_default_src = df_corte_p["_dt"]
    if df_par_p is not None and not df_par_p.empty:
        dt_default_src = pd.concat(
            [dt_default_src, df_par_p.get("dt_end", pd.Series([], dtype="datetime64[ns]"))],
            ignore_index=True,
        )

    if time_apply is None or str(time_apply).strip() == "":
        apply_time_filter = _should_apply_time_filter(time_start, time_end, dt_default_src)
    else:
        apply_time_filter = _to_bool(time_apply)

    if apply_time_filter:
        mw_corte = _apply_time_window(df_corte_p["_dt"], time_start, time_end)
        df_corte_w = df_corte_p.loc[mw_corte].copy()
    else:
        df_corte_w = df_corte_p.copy()

    # 9) KPIs Circuitos
    planeadas = 0
    cortados = 0
    no_conformes = 0

    if corte_cols.col_total and corte_cols.col_total in df_corte_w.columns:
        planeadas = int(pd.to_numeric(df_corte_w[corte_cols.col_total], errors="coerce").fillna(0).sum())
    if corte_cols.col_buenas and corte_cols.col_buenas in df_corte_w.columns:
        cortados = int(pd.to_numeric(df_corte_w[corte_cols.col_buenas], errors="coerce").fillna(0).sum())
    if corte_cols.col_solic and corte_cols.col_solic in df_corte_w.columns:
        no_conformes = int(pd.to_numeric(df_corte_w[corte_cols.col_solic], errors="coerce").fillna(0).sum())

    # 10) Defaults (si NO hay buckets)
    dead_total_sec = 0
    comida_sec = 0
    pausa_sec = 0
    otro_dead_sec = 0
    prod_sec = 0
    cap_total_sec = 0
    no_reg_sec_from_buckets = 0
    buckets: List[Dict[str, Any]] = []

    p = (period or "day").strip().lower()

    if p == "day":
        selected_day = pd.Timestamp(r0).normalize()

        buckets = _compute_hourly_buckets_hp(
            df_corte=df_corte_w,
            dt_corte=df_corte_w["_dt"],
            corte_cols=corte_cols,
            selected_day=selected_day,
            df_paradas_iv=df_par_work,
            time_start=time_start,
            time_end=time_end,
        )

        cap_total_sec = int(sum(int(b.get("capacitySec", 0) or 0) for b in buckets))

        # ✅ KPI efectivo = suma tarjetas
        prod_sec = int(sum(int(b.get("prodSec", 0) or 0) for b in buckets))

        # ✅ 105 y 104 (solo registrado)
        comida_sec = int(sum(int(b.get("mealSec", 0) or 0) for b in buckets))
        pausa_sec = int(sum(int(b.get("pauseSec", 0) or 0) for b in buckets))

        # ✅ KPI "Otro" = registrado excluyendo 105 y 104
        otro_dead_sec = int(sum(int(b.get("otherRecordedSec", 0) or 0) for b in buckets))

        # dead total mostrado tarjetas (incluye cierre 60 con 000)
        dead_total_sec = int(sum(int(b.get("deadSec", 0) or 0) for b in buckets))

        # 000 puro por tarjetas (debug)
        no_reg_sec_from_buckets = int(sum(int(b.get("unregSec", 0) or 0) for b in buckets))

    else:
        # week/month = sumando días usando la MISMA lógica de buckets por día
        def _iter_days(a: pd.Timestamp, b: pd.Timestamp):
            d0 = pd.Timestamp(a).normalize()
            d1 = pd.Timestamp(b).normalize()
            d = d0
            while d <= d1:
                yield d
                d += pd.Timedelta(days=1)

        tot_cap = tot_prod = tot_otro = tot_meal = tot_pause = tot_dead = tot_unreg = 0

        for day in _iter_days(r0, r1):
            day_start = pd.Timestamp(day).normalize()
            day_end = day_start + pd.Timedelta(days=1)

            m_c = df_corte_w["_dt"].notna() & (df_corte_w["_dt"] >= day_start) & (df_corte_w["_dt"] < day_end)
            df_corte_day = df_corte_w.loc[m_c].copy()

            df_par_day = None
            if df_par_work is not None and not df_par_work.empty:
                m_p = (
                        df_par_work["dt_end"].notna() & df_par_work["dt_start"].notna()
                        & (df_par_work["dt_end"] > day_start)
                        & (df_par_work["dt_start"] < day_end)
                )
                df_par_day = df_par_work.loc[m_p].copy()

            buckets_day = _compute_hourly_buckets_hp(
                df_corte=df_corte_day,
                dt_corte=df_corte_day["_dt"],
                corte_cols=corte_cols,
                selected_day=day_start,
                df_paradas_iv=df_par_day,
                time_start=time_start,
                time_end=time_end,
            )

            tot_cap += int(sum(int(b.get("capacitySec", 0) or 0) for b in buckets_day))
            tot_prod += int(sum(int(b.get("prodSec", 0) or 0) for b in buckets_day))
            tot_meal += int(sum(int(b.get("mealSec", 0) or 0) for b in buckets_day))
            tot_pause += int(sum(int(b.get("pauseSec", 0) or 0) for b in buckets_day))
            tot_otro += int(sum(int(b.get("otherRecordedSec", 0) or 0) for b in buckets_day))
            tot_dead += int(sum(int(b.get("deadSec", 0) or 0) for b in buckets_day))
            tot_unreg += int(sum(int(b.get("unregSec", 0) or 0) for b in buckets_day))

        cap_total_sec = int(tot_cap)
        prod_sec = int(tot_prod)
        comida_sec = int(tot_meal)
        pausa_sec = int(tot_pause)
        otro_dead_sec = int(tot_otro)
        dead_total_sec = int(tot_dead)
        no_reg_sec_from_buckets = int(tot_unreg)
        buckets = []  # week/month no muestran hourly

        # fallback si no hay paradas y sí hay duración (solo total muerto, sin desglose)
        if (cap_total_sec == 0 and prod_sec == 0 and (df_par_p is None or df_par_p.empty)) and use_duracion:
            dead_total_sec = int(dead_from_duracion_sec)
            otro_dead_sec = int(dead_total_sec)
            comida_sec = 0
            pausa_sec = 0

    # KPI Horas Disponibles (sin tocar THB)
    hd_sec = _compute_horas_disponibles_hp(
        period=p,
        r0=r0,
        r1=r1,
        df_corte_p=df_corte_p,
        df_par_p=df_par_work,  # ✅ usar recortadas
    )

    # ✅ No registrado FINAL (000): balance vs HD
    # No_reg = HD - (Efectivo + Otro_KPI)   [105 y 104 NO entran]
    # ✅ No registrado FINAL (000) BALANCEADO al redondeo del UI (2 decimales)
    # Para que visualmente se cumpla:
    #   Horas Disponibles == Tiempo Efectivo + Otro Tiempo + No registrado
    no_reg_sec = _balance_no_reg_sec_ui(int(hd_sec), int(prod_sec), int(otro_dead_sec))

    # =========================
    # % (sin contemplar 105 ni 104) - usando el MISMO redondeo del UI
    # base_ui = efectivo + otro_kpi + no_registrado  (en horas con 2 decimales)
    # =========================
    def _fmt_pct(pct: float) -> str:
        pr = round(float(pct), 1)
        if abs(pr - round(pr)) < 1e-9:
            return f"{int(round(pr))}%"
        return f"{pr:.1f}%"

    hd_h_ui = _sec_to_ui_hours_2(int(hd_sec))
    prod_h_ui = _sec_to_ui_hours_2(int(prod_sec))
    otro_h_ui = _sec_to_ui_hours_2(int(otro_dead_sec))
    noreg_h_ui = _sec_to_ui_hours_2(int(no_reg_sec))

    base_ui = prod_h_ui + otro_h_ui + noreg_h_ui
    if base_ui <= 0:
        pct_eff_s = _fmt_pct(0.0)
        pct_otro_s = _fmt_pct(0.0)
    else:
        pct_eff_s = _fmt_pct(100.0 * prod_h_ui / base_ui)
        pct_otro_s = _fmt_pct(100.0 * otro_h_ui / base_ui)

    # (si lo sigues usando en lógica posterior)
    base_no105104 = int(prod_sec) + int(otro_dead_sec) + int(no_reg_sec)

    # salida
    kpis = {
        "circuitos_cortados": cortados,
        "circuitos_planeados": planeadas,
        "circuitos_no_conformes": no_conformes,

        "tiempo_trabajo_sec": int(cap_total_sec),
        "tiempo_trabajo_hhmm": _seconds_to_hhmm(float(cap_total_sec)),

        "dead_total_sec": int(dead_total_sec),
        "dead_tiempo_hhmm": _seconds_to_hhmm(float(dead_total_sec)),

        # ✅ KPI "Otro" = registrado EXCLUYENDO 105 y 104
        "otro_tiempo_sec": int(otro_dead_sec),
        "otro_tiempo_hhmm": _seconds_to_hhmm(float(otro_dead_sec)),

        # ✅ 105 y ✅ 104 separados
        "comida_sec": int(comida_sec),
        "comida_hhmm": _seconds_to_hhmm(float(comida_sec)),

        "pausa_sec": int(pausa_sec),
        "pausa_hhmm": _seconds_to_hhmm(float(pausa_sec)),

        "tiempo_efectivo_sec": int(prod_sec),
        "tiempo_efectivo_hhmm": _format_hmm_from_seconds(int(prod_sec)),

        "horas_disponibles_sec": int(hd_sec),
        "horas_disponibles_hhmm": _format_hmm_from_seconds(int(hd_sec)),

        # ✅ No registrado FINAL (000)
        "no_registrado_sec": int(no_reg_sec),
        "no_registrado_hhmm": _seconds_to_hhmm(float(no_reg_sec)),

        # debug (000 por tarjetas)
        "no_registrado_from_buckets_sec": int(no_reg_sec_from_buckets),

        "hourly_buckets": buckets,
    }

    ui = {
        "Circuitos Cortados": f"{cortados:,}".replace(",", "."),
        "Circuitos Planeados": f"{planeadas:,}".replace(",", "."),
        "Circuitos No Conformes": f"{no_conformes:,}".replace(",", "."),

        "Horas Disponibles": kpis["horas_disponibles_hhmm"],

        # ✅ main = Otro KPI (sin 105/104/000)
        # ✅ sub = 105 + 104 + No registrado
        "Tiempos Perdidos": (
            f"{_seconds_to_hhmm(kpis['otro_tiempo_sec'])}"
            f"||105: {_seconds_to_hhmm(kpis['comida_sec'])}"
            f" | 104: {_seconds_to_hhmm(kpis['pausa_sec'])}"
            f" | No registrado: {_seconds_to_hhmm(kpis['no_registrado_sec'])}"
            f" | {pct_otro_s}"
        ),

        "Tiempo Trabajado": f"{kpis['tiempo_efectivo_hhmm']} ({pct_eff_s})",

    }

    pareto = _pareto_paradas_hp(df_par_work, top_n=25)

    return {
        "status": "ok",
        "analysis": ANALYSIS_KEY,
        "title": ANALYSIS_TITLE,
        "machine": machine,
        "period": period,
        "period_value": period_value,
        "operator": operator,
        "time_start": time_start,
        "time_end": time_end,
        "rows_total": int(len(df_corte_w)),
        "kpis": kpis,
        "kpis_ui": ui,
        "pareto": pareto,
    }







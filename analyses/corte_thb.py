# analyses/corte_thb.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import re

# ============================================================
#  Meta para registry
# ============================================================
ANALYSIS_KEY = "corte_thb"
ANALYSIS_TITLE = "Análisis Línea de Corte (THB)"

# ============================================================
# ✅ MODELO NOMINAL THB POR TIPO
# t(L_mm) = a + b*L_mm
# Tipo 1: 0 terminales
# Tipo 2: 1 terminal
# Tipo 3: 2 terminales
# ============================================================

THB_FIT_CALC = {
    1: {"a": 0.7011303800366301, "b": 0.0001308951465201466},
    2: {"a": 0.9724746750764526, "b": 0.00013737576452599423},
    3: {"a": 1.0754718802170284, "b": 0.00013864252921535892},
}

THB_FIT_V8 = {
    1: {"a": 0.7039453125, "b": 0.000125},
    2: {"a": 0.9777343750000002, "b": 0.000125},
    3: {"a": 1.0860069444444445, "b": 0.000125},
}


def _is_no_aplica(x: Any) -> bool:
    s = str(x or "").strip().lower()
    return (s == "") or (s == "no aplica") or (s == "na") or (s == "n/a") or (s == "none") or (s == "null")


def _thb_tipo_from_terminals(t1: Any, t2: Any) -> int:
    na1 = _is_no_aplica(t1)
    na2 = _is_no_aplica(t2)
    if na1 and na2:
        return 1
    if (na1 and not na2) or (not na1 and na2):
        return 2
    return 3


def _thb_nominal_sec(L_mm: float, tipo: int, *, model: str = "calc") -> float:
    if L_mm <= 0:
        return 0.0
    fit = (THB_FIT_CALC if model == "calc" else THB_FIT_V8).get(int(tipo), THB_FIT_CALC[1])
    a = float(fit["a"])
    b = float(fit["b"])
    t = a + b * float(L_mm)
    return float(max(t, 0.0))

# ============================================================
#  Utilidades robustas (fechas/horas/números)
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


def _excel_time_to_seconds(s: pd.Series) -> pd.Series:
    """
    Convierte Hora a segundos desde 00:00.
    - numérico: fracción de día Excel
    - string: "HH:MM" o "HH:MM:SS"
    """
    if s is None:
        return pd.Series([], dtype="float64")

    if pd.api.types.is_numeric_dtype(s):
        td = pd.to_timedelta(s, unit="D", errors="coerce")
        return td.dt.total_seconds()

    ss = s.astype(str).str.strip()
    td = pd.to_timedelta(ss, errors="coerce")
    return td.dt.total_seconds()


def _seconds_to_hhmm(sec: float) -> str:
    if sec is None or pd.isna(sec):
        return ""
    sec = int(max(0, float(sec)))
    hh = sec // 3600
    mm = (sec % 3600) // 60
    return f"{hh:02d}:{mm:02d}"


def _format_hmm_from_seconds(sec: float) -> str:
    if sec is None or pd.isna(sec) or sec < 0:
        return "0:00"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h}:{m:02d}"


def _num_es(x: Any) -> float:
    """
    Convierte "1.234,56" / "1234,56" / "1234.56" a float.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)

    s = str(x).strip().replace(" ", "")
    if not s:
        return 0.0

    lc = s.rfind(",")
    ld = s.rfind(".")
    if lc > -1 and ld > -1:
        if lc > ld:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif lc > -1:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")

    try:
        return float(s)
    except Exception:
        return 0.0


def _is_general_operator(op: str) -> bool:
    op = (op or "").strip().lower()
    return (op == "") or (op == "general")


def _time_str_to_seconds(t: str) -> int:
    s = (t or "").strip()
    if not s:
        return 0
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if not m:
        return 0
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3) or 0)
    return hh * 3600 + mm * 60 + ss


def _mask_time_window(dt: pd.Series, time_start: str, time_end: str) -> pd.Series:
    """
    Máscara por hora del día para datetimes.
    Soporta ventana normal (start<=end) o cruzando medianoche.
    """
    if dt is None or len(dt) == 0:
        return pd.Series([], dtype=bool)

    dtc = pd.to_datetime(dt, errors="coerce")
    s0 = _time_str_to_seconds(time_start)
    s1 = _time_str_to_seconds(time_end)

    sec = (dtc.dt.hour * 3600 + dtc.dt.minute * 60 + dtc.dt.second).astype("float64")

    if s0 <= s1:
        return dtc.notna() & (sec >= s0) & (sec <= s1)
    else:
        return dtc.notna() & ((sec >= s0) | (sec <= s1))


# ============================================================
#  Diccionario fijo: Código -> Descripción (THB)
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
    """
    Normaliza cualquier valor a un código tipo "101".
    Soporta: 101, 101.0, "101", "101 PARADAS PLANEADAS", etc.
    """
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
    # ✅ robusto: acepta "105", "105.0", "105 Almuerzo", etc.
    c = _code_to_str(code_str)
    return STOP_CODE_DESC.get(str(c).strip(), "")



# ============================================================
#  Parse columnas THB (Verificación)
# ============================================================
@dataclass
class THBCols:
    col_fecha: Optional[str]
    col_hora: Optional[str]
    col_nombre: Optional[str]
    col_piezas_buenas: Optional[str]
    col_total_piezas: Optional[str]
    col_long_mm: Optional[str]
    col_res_mm: Optional[str]
    col_arnes: Optional[str]
    col_consecutivo: Optional[str]
    col_terminal1: Optional[str]   # ✅ NUEVO
    col_terminal2: Optional[str]   # ✅ NUEVO


def _resolve_thb_cols(df: pd.DataFrame) -> THBCols:
    # Helper: fallback por posición Excel (A=0, B=1, ..., G=6, M=12, N=13)
    def _col_at(idx: int) -> Optional[str]:
        try:
            return df.columns[idx]
        except Exception:
            return None

    col_fecha = _find_col(df, ["fecha"])
    col_hora = _find_col(df, ["hora"])
    col_nombre = _find_col(df, ["nombre"])

    # ================================
    # FORZAR SEGÚN TU EXCEL:
    #   E (idx 4) = piezas planeadas
    #   G (idx 6) = piezas cortadas (buenas)
    #   M (idx 12)= longitud solicitada (mm)
    #   N (idx 13)= resultado verificación (mm)
    # ================================

    # Primero intenta por nombre (por si existe bien rotulado),
    # pero si falla, usa POSICIÓN fija (E/G/M/N) como fallback.
    col_total_piezas = (
        _find_col(df, ["total", "piezas"]) or _find_col(df, ["planead"]) or _col_at(4)  # E
    )
    col_piezas_buenas = (
        _find_col(df, ["piezas", "buenas"]) or _find_col(df, ["cortad"]) or _col_at(6)  # G
    )

    col_long_mm = (
        _find_col(df, ["longitud"]) or _find_col(df, ["solicit"]) or _col_at(12)  # M
    )
    col_res_mm = (
        _find_col(df, ["resultado"]) or _find_col(df, ["verific"]) or _col_at(13)  # N
    )

    col_arnes = _find_col(df, ["código", "arnés"]) or _find_col(df, ["arnes"])
    col_consecutivo = _find_col(df, ["consecutivo", "tarjeta"]) or _find_col(df, ["consecutivo"])

    col_terminal1 = _find_col(df, ["terminal", "1"]) or _col_at(7)  # H
    col_terminal2 = _find_col(df, ["terminal", "2"]) or _col_at(8)  # I

    return THBCols(
        col_fecha=col_fecha,
        col_hora=col_hora,
        col_nombre=col_nombre,
        col_piezas_buenas=col_piezas_buenas,
        col_total_piezas=col_total_piezas,
        col_long_mm=col_long_mm,
        col_res_mm=col_res_mm,
        col_arnes=col_arnes,
        col_consecutivo=col_consecutivo,
        col_terminal1=col_terminal1,
        col_terminal2=col_terminal2,
    )

def _fmt_lote_dt(x: Any) -> str:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _consecutivo_lote_key(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""

    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""

    try:
        n = float(s.replace(",", "."))
        if np.isfinite(n) and abs(n - round(n)) < 1e-9:
            return str(int(round(n)))
    except Exception:
        pass

    return s


def _build_thb_lotes_acumulados(
    df_verif: pd.DataFrame,
    dt_verif: pd.Series,
    cols: THBCols,
) -> List[Dict[str, Any]]:
    if df_verif is None or df_verif.empty or dt_verif is None or len(dt_verif) == 0:
        return []

    if not cols.col_arnes or cols.col_arnes not in df_verif.columns:
        return []

    if not cols.col_consecutivo or cols.col_consecutivo not in df_verif.columns:
        return []

    work = df_verif.copy()
    work["_dt_lote"] = pd.to_datetime(dt_verif, errors="coerce")
    work = work.loc[work["_dt_lote"].notna()].copy()
    if work.empty:
        return []

    work["_arnes_lote"] = work[cols.col_arnes].astype(str).str.strip()
    work["_cons_lote"] = work[cols.col_consecutivo].apply(_consecutivo_lote_key)

    work = work.loc[
        (work["_arnes_lote"] != "") &
        (work["_arnes_lote"].str.lower() != "nan") &
        (work["_cons_lote"] != "")
    ].copy()
    if work.empty:
        return []

    work = work.sort_values(["_dt_lote"], kind="stable").reset_index(drop=True)

    lotes_out = []
    lotes_activos_por_arnes = {}
    next_lote_global = 0

    def _nuevo_lote(arnes: str, cons: str, dt: pd.Timestamp):
        nonlocal next_lote_global

        next_lote_global += 1
        lote = {
            "lote": int(next_lote_global),
            "codigo_arnes": arnes,
            "consecutivo_inicio": cons,
            "consecutivo_fin": cons,
            "inicio_dt": dt,
            "fin_dt": dt,
            "tarjetas": 1,
            "vistos": {cons},
            "dias_set": {dt.date()},
        }
        lotes_activos_por_arnes[arnes] = lote
        lotes_out.append(lote)

    for _, row in work.iterrows():
        arnes = str(row["_arnes_lote"] or "").strip()
        cons = str(row["_cons_lote"] or "").strip()
        dt = pd.to_datetime(row["_dt_lote"], errors="coerce")

        if not arnes or not cons or pd.isna(dt):
            continue

        lote_activo = lotes_activos_por_arnes.get(arnes)

        # Si esa referencia nunca ha aparecido, crea su primer lote
        if lote_activo is None:
            _nuevo_lote(arnes, cons, dt)
            continue

        # Si el consecutivo ya había aparecido en el lote activo de ESA referencia,
        # entonces abre un nuevo lote para esa referencia.
        if cons in lote_activo["vistos"]:
            _nuevo_lote(arnes, cons, dt)
            continue

        # Si no se repite, sigue en el mismo lote,
        # aunque hayan pasado otros días o se hayan procesado otras referencias entre medio.
        lote_activo["vistos"].add(cons)
        lote_activo["consecutivo_fin"] = cons
        lote_activo["fin_dt"] = dt
        lote_activo["tarjetas"] += 1
        lote_activo["dias_set"].add(dt.date())

    # salida final limpia
    out = []
    for lote in sorted(lotes_out, key=lambda x: x["inicio_dt"]):
        inicio_dt = pd.to_datetime(lote["inicio_dt"], errors="coerce")
        fin_dt = pd.to_datetime(lote["fin_dt"], errors="coerce")

        dur_sec = 0
        if pd.notna(inicio_dt) and pd.notna(fin_dt):
            dur_sec = int(max(0, (fin_dt - inicio_dt).total_seconds()))

        out.append({
            "lote": int(lote["lote"]),
            "codigo_arnes": str(lote["codigo_arnes"]),
            "consecutivo_inicio": str(lote["consecutivo_inicio"]),
            "consecutivo_fin": str(lote["consecutivo_fin"]),
            "inicio": _fmt_lote_dt(inicio_dt),
            "fin": _fmt_lote_dt(fin_dt),
            "duracion_hhmm": _format_hmm_from_seconds(dur_sec),
            "tarjetas": int(lote["tarjetas"]),
            "dias": int(len(lote["dias_set"])),
        })

    return out

def _coerce_excel_date_series(s: pd.Series) -> pd.Series:
    """
    Convierte una columna Fecha a datetime normalizado (00:00),
    soportando:
    - datetime ya parseado
    - serial Excel (numérico o string numérico)
    - strings ISO (YYYY-MM-DD o YYYY/MM/DD)
    - strings dd/mm/yyyy (Colombia, dayfirst=True)
    """
    if s is None:
        return pd.Series([], dtype="datetime64[ns]")

    # ya viene como datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce").dt.normalize()

    # serial Excel numérico
    if pd.api.types.is_numeric_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce").dt.normalize()

    # strings / mixto
    txt = s.astype(str).str.strip()
    txt = txt.replace({"": None, "nan": None, "None": None, "NaT": None})

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # 1) ISO yyyy-mm-dd / yyyy/mm/dd (no ambigua)
    mask_iso = txt.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", na=False)
    if mask_iso.any():
        out.loc[mask_iso] = pd.to_datetime(
            txt.loc[mask_iso].str.replace("/", "-", regex=False),
            errors="coerce",
            yearfirst=True,
            dayfirst=False,
        )

    # 2) serial Excel pero como texto ("45234")
    num = pd.to_numeric(txt, errors="coerce")
    mask_serial = num.notna() & out.isna()
    if mask_serial.any():
        out.loc[mask_serial] = pd.to_datetime(
            num.loc[mask_serial],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    # 3) dd/mm/yyyy (preferido Colombia) — aquí resolvemos la ambigüedad <=12
    mask_dmy = out.isna() & txt.str.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", na=False)
    if mask_dmy.any():
        out.loc[mask_dmy] = pd.to_datetime(txt.loc[mask_dmy], errors="coerce", dayfirst=True)

    # 4) fallback final
    rem = out.isna()
    if rem.any():
        out.loc[rem] = pd.to_datetime(txt.loc[rem], errors="coerce", dayfirst=True)

    return out.dt.normalize()



def _build_dt(df: pd.DataFrame, col_fecha: Optional[str], col_hora: Optional[str]) -> pd.Series:
    if not col_fecha or col_fecha not in df.columns:
        return pd.Series([], dtype="datetime64[ns]")

    dt_fecha = _coerce_excel_date_series(df[col_fecha])

    if not col_hora or col_hora not in df.columns:
        return dt_fecha

    sec = _excel_time_to_seconds(df[col_hora]).fillna(0)
    return dt_fecha + pd.to_timedelta(sec, unit="s")


# ============================================================
#  Paradas (opcional)
# ============================================================
@dataclass
class ParadasCols:
    col_fecha: Optional[str]
    col_hora: Optional[str]
    col_codigo: Optional[str]
    col_duracion: Optional[str]
    col_desc: Optional[str]


def _resolve_paradas_cols(df: pd.DataFrame) -> ParadasCols:
    col_fecha = _find_col(df, ["fecha"])
    col_hora = _find_col(df, ["hora"])
    col_codigo = _find_col(df, ["código"]) or _find_col(df, ["codigo"])
    col_duracion = _find_col(df, ["duración"]) or _find_col(df, ["duracion"])
    col_desc = _find_col(df, ["descrip"]) or _find_col(df, ["descripción"]) or _find_col(df, ["descripcion"])
    return ParadasCols(col_fecha, col_hora, col_codigo, col_duracion, col_desc)


def _build_paradas_intervals(df_paradas: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve df con columnas:
      - dt_end (datetime)   <- hora de FINALIZACIÓN de la parada
      - dt_start (datetime) <- dt_end - dur
      - codigo (str)        <- normalizado
      - desc (str)          <- descripción asociada al código (si existe)
      - dur_sec (float)
    """
    cols = _resolve_paradas_cols(df_paradas)
    if not cols.col_fecha or not cols.col_hora or not cols.col_duracion:
        return pd.DataFrame(columns=["dt_start", "dt_end", "codigo", "desc", "dur_sec"])

    dt_end = _build_dt(df_paradas, cols.col_fecha, cols.col_hora)

    dur = df_paradas[cols.col_duracion]
    if pd.api.types.is_numeric_dtype(dur):
        dur_sec = pd.to_numeric(dur, errors="coerce").fillna(0).astype(float)
    else:
        dur_sec = pd.to_numeric(dur.astype(str).str.strip(), errors="coerce").fillna(0).astype(float)

    dt_start = dt_end - pd.to_timedelta(dur_sec, unit="s")

    if cols.col_codigo and cols.col_codigo in df_paradas.columns:
        codigo = df_paradas[cols.col_codigo].apply(_code_to_str)
    else:
        codigo = pd.Series([""] * len(df_paradas), index=df_paradas.index)

    if cols.col_desc and cols.col_desc in df_paradas.columns:
        desc = df_paradas[cols.col_desc].astype(str).str.strip()
        desc = desc.where(~desc.str.lower().isin(["nan", "none"]), "")
    else:
        desc = pd.Series([""] * len(df_paradas), index=df_paradas.index)

    out = pd.DataFrame(
        {
            "dt_start": dt_start,
            "dt_end": dt_end,
            "codigo": codigo,
            "desc": desc,
            "dur_sec": dur_sec,
        }
    )

    out = out.dropna(subset=["dt_start", "dt_end"])

    out["codigo"] = out["codigo"].astype(str).str.strip()
    out["desc"] = out["desc"].astype(str).str.strip()
    out["dur_sec"] = pd.to_numeric(out["dur_sec"], errors="coerce").fillna(0.0).astype(float)

    out = out[(out["codigo"] != "") & (out["dur_sec"] > 0)].copy()

    if "desc" in out.columns and not out.empty:
        out["desc"] = (
            out.groupby("codigo", dropna=False)["desc"]
            .transform(lambda x: x.replace("", pd.NA).ffill().bfill().fillna(""))
        )
    # ✅ si sigue vacío, fuerza desde diccionario fijo (clave para 105)
    if "desc" in out.columns and not out.empty:
        mapped = out["codigo"].astype(str).map(lambda x: _desc_from_dict(x))
        out["desc"] = out["desc"].where(out["desc"].astype(str).str.strip() != "", mapped)


    return out


def _clip_intervals_to_range(df_iv: pd.DataFrame, r0: pd.Timestamp, r1: pd.Timestamp) -> pd.DataFrame:
    """
    Recorta intervalos [dt_start, dt_end] al rango [r0, r1) y recalcula dur_sec.
    Esto evita que una parada que cruza medianoche infle el total.
    """
    if df_iv is None or df_iv.empty:
        return df_iv
    if ("dt_start" not in df_iv.columns) or ("dt_end" not in df_iv.columns):
        return df_iv

    d = df_iv.copy()
    d["dt_start"] = pd.to_datetime(d["dt_start"], errors="coerce")
    d["dt_end"] = pd.to_datetime(d["dt_end"], errors="coerce")
    d = d.dropna(subset=["dt_start", "dt_end"])
    if d.empty:
        return d

    s = d["dt_start"].where(d["dt_start"] > r0, r0)
    e = d["dt_end"].where(d["dt_end"] < r1, r1)

    dur = (e - s).dt.total_seconds()
    dur = pd.to_numeric(dur, errors="coerce").fillna(0.0).clip(lower=0.0)

    d["dur_sec"] = dur.astype(float)
    d = d[d["dur_sec"] > 0].copy()
    return d


def _filter_paradas_by_operator_using_verif(
    df_paradas_iv: pd.DataFrame,
    df_verif: pd.DataFrame,
    dt_verif: pd.Series,
    cols_verif: THBCols,
    operator: str,
) -> pd.DataFrame:
    """
    Asigna paradas a operaria usando timeline de Verificación (merge_asof).
    ✅ FIX: si en el periodo/día solo existe UNA operaria, entonces seleccionar esa operaria
    debe devolver EXACTAMENTE lo mismo que "General" (no debe perder paradas tempranas).
    """
    if df_paradas_iv is None or df_paradas_iv.empty:
        return df_paradas_iv
    if _is_general_operator(operator):
        return df_paradas_iv
    if not cols_verif.col_nombre or cols_verif.col_nombre not in df_verif.columns:
        return df_paradas_iv

    sel = str(operator).strip()

    # Timeline de operaria desde Verificación
    v = pd.DataFrame({
        "dt": dt_verif,
        "op": df_verif[cols_verif.col_nombre].astype(str).str.strip(),
    }).dropna(subset=["dt"])

    v = v[v["op"].astype(str).str.strip().ne("")].copy()
    if v.empty:
        return df_paradas_iv

    # ✅ SI SOLO HAY 1 OPERARIA EN EL PERIODO, NO FILTRES PARADAS:
    uniq_ops = [o for o in v["op"].dropna().astype(str).str.strip().unique().tolist() if o]
    if len(uniq_ops) == 1 and uniq_ops[0] == sel:
        return df_paradas_iv

    v = v.sort_values("dt")

    # conservar índice real de la parada
    p = df_paradas_iv[["dt_end"]].copy()
    p["p_idx"] = df_paradas_iv.index.to_numpy()
    p = p.dropna(subset=["dt_end"]).sort_values("dt_end")

    merged = pd.merge_asof(
        p,
        v,
        left_on="dt_end",
        right_on="dt",
        direction="backward",
        allow_exact_matches=True,
    )

    merged["op"] = merged["op"].astype(str).str.strip()

    keep_idx = merged.loc[merged["op"] == sel, "p_idx"]
    if keep_idx.empty:
        return df_paradas_iv.iloc[0:0].copy()

    keep_idx = keep_idx.dropna().astype(int).unique().tolist()
    return df_paradas_iv.loc[df_paradas_iv.index.isin(keep_idx)].copy()



# ============================================================
#  Period helpers
# ============================================================
def _normalize_period_value(period: str, period_value: str) -> str:
    p = (period or "").strip().lower()
    s = (period_value or "").strip()
    if not s:
        return ""

    s2 = s.replace("\\", "/").replace(".", "/").strip()

    if p == "day":
        if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", s2):
            dt = pd.to_datetime(s2.replace("/", "-"), errors="coerce", yearfirst=True, dayfirst=False)
            return dt.strftime("%Y-%m-%d") if not pd.isna(dt) else ""
        dt = pd.to_datetime(s2, errors="coerce", dayfirst=True)
        return dt.strftime("%Y-%m-%d") if not pd.isna(dt) else ""

    if p == "month":
        m = re.match(r"^(\d{4})[-/](\d{1,2})$", s2)
        if m:
            y = int(m.group(1))
            mo = int(m.group(2))
            if 1 <= mo <= 12:
                return f"{y:04d}-{mo:02d}"
            return ""
        m = re.match(r"^(\d{1,2})[-/](\d{4})$", s2)
        if m:
            mo = int(m.group(1))
            y = int(m.group(2))
            if 1 <= mo <= 12:
                return f"{y:04d}-{mo:02d}"
            return ""
        dt = pd.to_datetime(s2, errors="coerce", dayfirst=True)
        return dt.strftime("%Y-%m") if not pd.isna(dt) else ""

    if p == "week":
        m = re.match(r"^(\d{4})-?[Ww](\d{1,2})$", s2.replace(" ", ""))
        if m:
            y = int(m.group(1))
            w = int(m.group(2))
            if 1 <= w <= 53:
                return f"{y:04d}-W{w:02d}"
            return ""
        dt = pd.to_datetime(s2, errors="coerce", dayfirst=True)
        return dt.strftime("%Y-%m-%d") if not pd.isna(dt) else ""

    dt = pd.to_datetime(s2, errors="coerce", dayfirst=True)
    return dt.strftime("%Y-%m-%d") if not pd.isna(dt) else ""


def _period_bounds(period: str, pv: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    p = (period or "").strip().lower()
    pv = (pv or "").strip()
    if not pv:
        return None, None

    if p == "day":
        d = pd.to_datetime(pv, errors="coerce")
        if pd.isna(d):
            return None, None
        r0 = pd.Timestamp(d).normalize()
        r1 = r0 + pd.Timedelta(days=1)
        return r0, r1

    if p == "month":
        m = re.match(r"^(\d{4})-(\d{2})$", pv)
        if not m:
            d = pd.to_datetime(pv, errors="coerce", dayfirst=True)
            if pd.isna(d):
                return None, None
            r0 = pd.Timestamp(d).normalize().replace(day=1)
        else:
            y = int(m.group(1))
            mo = int(m.group(2))
            r0 = pd.Timestamp(year=y, month=mo, day=1)
        r1 = (r0 + pd.offsets.MonthBegin(1)).normalize()
        return r0, r1

    if p == "week":
        m = re.match(r"^(\d{4})-W(\d{2})$", pv)
        if m:
            y = int(m.group(1))
            w = int(m.group(2))
            try:
                r0 = pd.to_datetime(f"{y}-W{w:02d}-1", format="%G-W%V-%u", errors="coerce")
            except Exception:
                r0 = pd.NaT
            if pd.isna(r0):
                return None, None
            r0 = pd.Timestamp(r0).normalize()
            r1 = r0 + pd.Timedelta(days=7)
            return r0, r1

        d = pd.to_datetime(pv, errors="coerce", dayfirst=True)
        if pd.isna(d):
            return None, None
        d = pd.Timestamp(d).normalize()
        r0 = d - pd.Timedelta(days=int(d.weekday()))
        r1 = r0 + pd.Timedelta(days=7)
        return r0, r1

    return None, None


def _quantize_hour_parts_to_60min(prod_sec: float, meal_sec: float, other_sec: float) -> Tuple[int, int, int]:
    """
    Convierte a minutos (múltiplos de 60s) y fuerza que:
      prod + meal + other = 3600
    Retorna (prod_q, meal_q, other_q) en segundos.
    """
    # segundos -> minutos redondeados
    pm = int(round(max(0.0, float(prod_sec)) / 60.0))
    mm = int(round(max(0.0, float(meal_sec)) / 60.0))
    om = int(round(max(0.0, float(other_sec)) / 60.0))

    # clamps básicos
    pm = max(0, min(60, pm))
    mm = max(0, min(60, mm))
    om = max(0, min(60, om))

    # fuerza suma a 60 ajustando "other" como buffer
    om = 60 - pm - mm

    # si quedó negativo, recorta prod primero, luego meal
    if om < 0:
        deficit = -om
        take = min(deficit, pm)
        pm -= take
        deficit -= take
        if deficit > 0:
            take2 = min(deficit, mm)
            mm -= take2
            deficit -= take2
        om = 60 - pm - mm

    # clamp final
    pm = max(0, min(60, pm))
    mm = max(0, min(60, mm))
    om = max(0, min(60, om))

    return pm * 60, mm * 60, om * 60

def _thb_paid_seconds_for_hour(hour_int: int) -> int:
    """
    Tiempo pagado por franja horaria THB.
    Reglas solicitadas:
      - 08:00–09:00 => 45 min pagados
      - 12:00–13:00 => 30 min pagados
      - resto => 60 min pagados
    """
    h = int(hour_int)

    if h == 8:
        return 45 * 60   # 0.75 h

    if h == 13:
        return 30 * 60  # 13:00–14:00 = 0.50 h

    return 60 * 60

# ============================================================
#  Overlap helpers
# ============================================================
def _overlap_seconds(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> float:
    start = max(a0.value, b0.value)
    end = min(a1.value, b1.value)
    return max(0.0, (end - start) / 1e9)


def _sum_overlap_seconds(df_iv: pd.DataFrame, w0: pd.Timestamp, w1: pd.Timestamp, code: Optional[str] = None) -> int:
    if df_iv is None or df_iv.empty:
        return 0
    if "dt_start" not in df_iv.columns or "dt_end" not in df_iv.columns:
        return 0

    d = df_iv.copy()
    d["dt_start"] = pd.to_datetime(d["dt_start"], errors="coerce")
    d["dt_end"] = pd.to_datetime(d["dt_end"], errors="coerce")
    d = d.dropna(subset=["dt_start", "dt_end"])
    if d.empty:
        return 0

    if code is not None and "codigo" in d.columns:
        cc = d["codigo"].astype(str).str.strip()
        d = d.loc[cc.eq(str(code).strip())].copy()
        if d.empty:
            return 0

    s = d["dt_start"].where(d["dt_start"] > w0, w0)
    e = d["dt_end"].where(d["dt_end"] < w1, w1)

    ov = (e - s).dt.total_seconds()
    ov = pd.to_numeric(ov, errors="coerce").fillna(0.0).clip(lower=0.0)
    return int(ov.sum())







def _attach_other_causes_to_buckets_thb(
    buckets_in: List[Dict[str, Any]],
    df_paradas_iv: Optional[pd.DataFrame],
    selected_day: pd.Timestamp,
    *,
    top_n: int = 10,
) -> None:
    """
    Agrega a cada bucket:
      bucket["other_causes"] = [
        {"code": "203", "seconds": 420, "hhmm": "00:07", "desc": "..."},
        {"code": "203", "seconds": 180, "hhmm": "00:03", "desc": "..."},
        ...
      ]

    ✅ Cambio: NO acumula por código. Si hay 2+ paradas del mismo código en la hora,
              se muestran como entradas separadas (una por evento).
    """
    if not buckets_in:
        return

    for b in buckets_in:
        b["other_causes"] = []

    if df_paradas_iv is None or not isinstance(df_paradas_iv, pd.DataFrame) or df_paradas_iv.empty:
        return
    if ("dt_start" not in df_paradas_iv.columns) or ("dt_end" not in df_paradas_iv.columns):
        return

    MEAL_CODE = "105"

    day0 = pd.Timestamp(selected_day).normalize()
    day_iso = day0.strftime("%Y-%m-%d")
    day_start = pd.Timestamp(f"{day_iso} 00:00:00")
    day_end = pd.Timestamp(f"{day_iso} 23:59:59.999")

    p = df_paradas_iv.copy()
    p["dt_start"] = pd.to_datetime(p["dt_start"], errors="coerce")
    p["dt_end"] = pd.to_datetime(p["dt_end"], errors="coerce")
    p = p.dropna(subset=["dt_start", "dt_end"])
    if p.empty:
        return

    # recorta al día
    p = p.loc[(p["dt_end"] >= day_start) & (p["dt_start"] <= day_end)].copy()
    if p.empty:
        return

    p.loc[p["dt_start"] < day_start, "dt_start"] = day_start
    p.loc[p["dt_end"] > day_end, "dt_end"] = day_end
    p = p[p["dt_end"] > p["dt_start"]].copy()
    if p.empty:
        return

    # códigos normalizados
    p["codigo"] = p.get("codigo", "").apply(_code_to_str).astype(str).str.strip()
    p = p[p["codigo"].ne("")].copy()
    if p.empty:
        return

    # mapa desc por código (prefiere columna desc si existe)
    desc_map: Dict[str, str] = {}
    if "desc" in p.columns:
        tmp = p[["codigo", "desc"]].copy()
        tmp["desc"] = tmp["desc"].astype(str).str.strip()
        tmp.loc[tmp["desc"].str.lower().isin(["nan", "none"]), "desc"] = ""
        tmp = tmp[tmp["desc"].ne("")]
        if not tmp.empty:
            for c, grp in tmp.groupby("codigo", dropna=False):
                v = grp["desc"].iloc[0]
                if v:
                    desc_map[str(c).strip()] = str(v)

    def _get_desc(code: str) -> str:
        code = str(code).strip()
        if code == "000":
            return "Desconocido"
        if code in desc_map and desc_map[code]:
            return desc_map[code]
        return _desc_from_dict(code) or ""

    # arrays ns para overlap rápido
    start_ns = p["dt_start"].astype("datetime64[ns]").astype("int64").to_numpy()
    end_ns = p["dt_end"].astype("datetime64[ns]").astype("int64").to_numpy()
    codes_np = p["codigo"].astype(str).to_numpy()

    for b in buckets_in:
        h_raw = str(b.get("hour") or b.get("hourLabel") or "").strip()
        h_txt = h_raw.split(":")[0] if h_raw else ""
        try:
            h = int(h_txt)
        except Exception:
            b["other_causes"] = []
            continue

        h0 = pd.Timestamp(f"{day_iso} {h:02d}:00:00")
        h1 = h0 + pd.Timedelta(hours=1)

        h0_ns = int(h0.value)
        h1_ns = int(h1.value)

        # overlap por fila (evento)
        ol_ns = np.minimum(end_ns, h1_ns) - np.maximum(start_ns, h0_ns)
        ol_ns = np.maximum(0, ol_ns)

        if ol_ns.sum() <= 0:
            b["other_causes"] = []
            continue

        # ✅ en vez de acumular por código, creamos un item por evento
        events_min: List[Tuple[str, int]] = []  # (codigo, minutos)
        for code, ns in zip(codes_np, ol_ns):
            cc = str(code).strip()
            if (not cc) or (cc == MEAL_CODE):
                continue
            sec = int(ns / 1e9)
            if sec <= 0:
                continue

            # cuantiza cada evento a minutos
            m = int(round(sec / 60.0))
            if m <= 0:
                m = 1  # si hubo solape, al menos 1 min para que se vea
            events_min.append((cc, m))

        # unknown (del bucket) también entra como evento "000"
        unk = int(b.get("unknownSec") or 0)
        if unk > 0:
            m_unk = int(round(unk / 60.0))
            if m_unk > 0:
                events_min.append(("000", m_unk))

        if not events_min:
            b["other_causes"] = []
            continue

        # ✅ ajustar para que sume EXACTO lo que dice otherDeadSec del bucket
        target_sec = int(b.get("otherDeadSec") or 0)
        target_min = max(0, target_sec // 60)

        sum_min = sum(m for _, m in events_min)
        delta = target_min - sum_min

        if delta != 0:
            # preferimos ajustar el "000" (Desconocido). Si no existe, ajusta el último evento.
            idx = None
            for i in range(len(events_min) - 1, -1, -1):
                if events_min[i][0] == "000":
                    idx = i
                    break
            if idx is None:
                idx = len(events_min) - 1

            c0, m0 = events_min[idx]
            m1 = max(0, m0 + delta)
            events_min[idx] = (c0, m1)

            # si quedó en 0, lo eliminamos
            events_min = [(c, m) for (c, m) in events_min if m > 0]

        # ordenar por duración desc
        events_min.sort(key=lambda x: x[1], reverse=True)

        # top_n opcional (si quieres ver TODO, pon top_n=0 o un número muy grande)
        if top_n and len(events_min) > top_n:
            keep_n = max(1, top_n - 1)
            keep = events_min[:keep_n]
            rest_min = sum(m for _, m in events_min[keep_n:])
            keep.append(("999", rest_min))
            events_min = keep

        # salida final (min -> sec)
        b["other_causes"] = [
            {
                "code": c,
                "seconds": int(m * 60),
                "hhmm": _seconds_to_hhmm(float(m * 60)),
                "desc": ("Otros" if c == "999" else _get_desc(c)),
            }
            for c, m in events_min
        ]


# ============================================================
#  Productividad por hora
# ============================================================
def _compute_hourly_buckets(
    df_verif: pd.DataFrame,
    dt_verif: pd.Series,
    cols_verif: THBCols,
    selected_day: pd.Timestamp,
    df_paradas_iv: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    Retorna lista de buckets por hora:
      - prodSec, deadSec, mealSec, otherDeadSec, cut, meters, hadActivity, capacitySec
      + ✅ THB: TC Nominal Ponderado por hora (tcnpSec), longitud ponderada (tcnpLenMm), total nominal (tcnpTotalSec)
    """
    LEN_SMALL_MAX = 300.0
    LEN_MED_MAX = 800.0
    MEAL_CODE = "105"

    day_iso = selected_day.strftime("%Y-%m-%d")
    day_start = pd.Timestamp(f"{day_iso} 00:00:00")
    day_end = pd.Timestamp(f"{day_iso} 23:59:59.999")

    times: List[pd.Timestamp] = []
    if dt_verif is not None and not dt_verif.dropna().empty:
        vts = pd.to_datetime(dt_verif.dropna(), errors="coerce")
        vts = vts[(vts >= day_start) & (vts <= day_end)]
        times += vts.tolist()

    stop_intervals: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if df_paradas_iv is not None and not df_paradas_iv.empty:
        for _, r in df_paradas_iv.iterrows():
            s = pd.to_datetime(r.get("dt_start"), errors="coerce")
            e = pd.to_datetime(r.get("dt_end"), errors="coerce")
            if pd.isna(s) or pd.isna(e):
                continue
            if e < day_start or s > day_end:
                continue
            if s < day_start:
                s = day_start
            if e > day_end:
                e = day_end
            code = str(r.get("codigo", "")).strip()
            stop_intervals.append((s, e, code))
            times.append(s)
            times.append(e)

    if not times:
        return []

    activity_min = min(times)
    activity_max = max(times)

    c_buenas = cols_verif.col_piezas_buenas
    c_total = cols_verif.col_total_piezas
    c_resmm = cols_verif.col_res_mm
    c_longmm = cols_verif.col_long_mm

    buckets: List[Dict[str, Any]] = []

    for h in range(24):
        h0 = pd.Timestamp(f"{day_iso} {h:02d}:00:00")
        h1 = h0 + pd.Timedelta(hours=1)
        paid_sec = int(_thb_paid_seconds_for_hour(h))

        cap_raw = _overlap_seconds(h0, h1, activity_min, activity_max)
        if cap_raw <= 0:
            continue

        if dt_verif is not None:
            dtc = pd.to_datetime(dt_verif, errors="coerce")
            in_hour = dtc.notna() & (dtc >= h0) & (dtc < h1)
        else:
            in_hour = pd.Series([], dtype=bool)

        sub = df_verif.loc[in_hour].copy() if len(in_hour) else df_verif.iloc[0:0].copy()

        cut = 0
        meters = 0.0

        tcnp_sec = 0.0
        tcnp_len_mm = 0.0
        tcnp_total_sec = 0.0

        had_corte = False

        if not sub.empty:
            if c_buenas and c_buenas in sub.columns:
                cut = int(pd.to_numeric(sub[c_buenas], errors="coerce").fillna(0).sum())

            if c_buenas and c_resmm and (c_buenas in sub.columns) and (c_resmm in sub.columns):
                buenas = pd.to_numeric(sub[c_buenas], errors="coerce").fillna(0).astype(float)
                resmm = sub[c_resmm].apply(_num_es).astype(float)

                meters = float(((buenas * resmm) / 1000.0).sum())

                q_total = float(buenas.sum())
                sum_q_mm = float((buenas * resmm).sum())

                if q_total > 0 and sum_q_mm > 0:
                    tcnp_len_mm = sum_q_mm / q_total

                c_t1 = getattr(cols_verif, "col_terminal1", None)
                c_t2 = getattr(cols_verif, "col_terminal2", None)

                if q_total > 0:
                    if c_t1 and c_t2 and (c_t1 in sub.columns) and (c_t2 in sub.columns):
                        tipos = sub.apply(lambda r: _thb_tipo_from_terminals(r.get(c_t1), r.get(c_t2)), axis=1)
                    else:
                        tipos = pd.Series([1] * len(sub), index=sub.index)

                    t_nom = [
                        _thb_nominal_sec(float(L), int(tp), model="calc")
                        for L, tp in zip(resmm.tolist(), tipos.tolist())
                    ]
                    t_nom = pd.Series(t_nom, index=sub.index, dtype="float64")

                    tcnp_total_sec = float((buenas * t_nom).sum())
                    tcnp_sec = float(tcnp_total_sec / q_total) if q_total > 0 else 0.0

            had_corte = (cut > 0) or (meters > 0)

            if c_longmm and (c_longmm in sub.columns):
                longmm = sub[c_longmm].apply(_num_es)
                if c_total and (c_total in sub.columns):
                    tot = pd.to_numeric(sub[c_total], errors="coerce").fillna(0).astype(float)
                else:
                    tot = pd.Series([0.0] * len(sub), index=sub.index, dtype="float64")

                if c_buenas and (c_buenas in sub.columns):
                    buenas2 = pd.to_numeric(sub[c_buenas], errors="coerce").fillna(0).astype(float)
                else:
                    buenas2 = pd.Series([0.0] * len(sub), index=sub.index, dtype="float64")

                w = tot.where(tot > 0, buenas2)
                w = w.where(w > 0, 1.0)

                valid = pd.to_numeric(longmm, errors="coerce").notna() & (longmm > 0)
                longmm = longmm.where(valid)
                w = w.where(valid).fillna(0.0)

                tot_w = float(w.sum())
                if tot_w > 0:
                    mask_small = (longmm <= LEN_SMALL_MAX)
                    mask_med = (longmm > LEN_SMALL_MAX) & (longmm <= LEN_MED_MAX)
                    mask_long = (longmm > LEN_MED_MAX)

                    w_small = float(w.where(mask_small, 0.0).sum())
                    w_med = float(w.where(mask_med, 0.0).sum())
                    w_long = float(w.where(mask_long, 0.0).sum())

                    sub_len_mix = {
                        "short_pct": round((w_small / tot_w) * 100.0),
                        "mid_pct": round((w_med / tot_w) * 100.0),
                        "long_pct": round((w_long / tot_w) * 100.0),
                    }
                else:
                    sub_len_mix = None
            else:
                sub_len_mix = None
        else:
            sub_len_mix = None

        dead_stop = 0.0
        meal_stop = 0.0
        had_parada = False

        for (s, e, code) in stop_intervals:
            olap = _overlap_seconds(h0, h1, s, e)
            if olap > 0:
                dead_stop += olap
                had_parada = True
                if str(code).strip() == MEAL_CODE:
                    meal_stop += olap

        cap_raw = float(cap_raw)
        if cap_raw < 0:
            cap_raw = 0.0
        if cap_raw > 3600.0:
            cap_raw = 3600.0

        unknown_raw = max(0.0, float(paid_sec) - cap_raw)

        fixed_unpaid_sec = 0.0
        if h == 8:
            fixed_unpaid_sec = 15 * 60
        elif h == 13:
            fixed_unpaid_sec = 30 * 60

        meal_stop_adj = max(0.0, float(meal_stop) - fixed_unpaid_sec)

        dead_other_raw = max(0.0, float(dead_stop) - float(meal_stop))
        other_raw = dead_other_raw + unknown_raw

        prod_raw = max(0.0, cap_raw - float(dead_stop) + fixed_unpaid_sec)

        if prod_raw > float(paid_sec):
            prod_raw = float(paid_sec)

        if meal_stop_adj > float(paid_sec):
            meal_stop_adj = float(paid_sec)

        if cut <= 0:
            prod_raw = 0.0
            other_raw = max(0.0, float(paid_sec) - float(meal_stop_adj))

        paid_min = max(0, int(round(float(paid_sec) / 60.0)))

        pm = int(round(max(0.0, float(prod_raw)) / 60.0))
        mm = int(round(max(0.0, float(meal_stop_adj)) / 60.0))
        om = int(round(max(0.0, float(other_raw)) / 60.0))

        pm = max(0, min(paid_min, pm))
        mm = max(0, min(paid_min, mm))
        om = max(0, min(paid_min, om))

        om = paid_min - pm - mm

        if om < 0:
            deficit = -om
            take = min(deficit, pm)
            pm -= take
            deficit -= take

            if deficit > 0:
                take2 = min(deficit, mm)
                mm -= take2
                deficit -= take2

            om = paid_min - pm - mm

        pm = max(0, min(paid_min, pm))
        mm = max(0, min(paid_min, mm))
        om = max(0, min(paid_min, om))

        prod_q = pm * 60
        meal_q = mm * 60
        other_q = om * 60

        dead_other_q = int(round(dead_other_raw / 60.0)) * 60
        if dead_other_q < 0:
            dead_other_q = 0
        if dead_other_q > other_q:
            dead_other_q = other_q

        unknown_q = int(other_q - dead_other_q)

        had_activity = bool(had_corte or had_parada)

        bucket = {
            "hourLabel": f"{h:02d}:00",
            "hour": f"{h:02d}",
            "capacitySec": int(paid_sec),
            "prodSec": int(prod_q),
            "mealSec": int(meal_q),
            "otherDeadSec": int(other_q),
            "deadSec": int(meal_q + other_q),
            "unknownSec": int(unknown_q),
            "cut": int(cut),
            "meters": float(meters),
            "tcnpSec": float(tcnp_sec),
            "tcnpLenMm": float(tcnp_len_mm),
            "tcnpTotalSec": float(tcnp_total_sec),
            "hadActivity": bool(had_activity),
        }
        if sub_len_mix is not None:
            bucket["len_mix"] = sub_len_mix

        buckets.append(bucket)

    return buckets





def _compute_prod_seconds_for_range(
    df_verif: pd.DataFrame,
    dt_verif: pd.Series,
    cols: THBCols,
    df_paradas_iv: Optional[pd.DataFrame],
) -> int:
    if (df_verif is None or df_verif.empty) and (df_paradas_iv is None or df_paradas_iv.empty):
        return 0

    days = set()
    if dt_verif is not None and not dt_verif.dropna().empty:
        days |= set(pd.to_datetime(dt_verif.dropna()).dt.normalize().tolist())
    if df_paradas_iv is not None and not df_paradas_iv.empty and "dt_end" in df_paradas_iv.columns:
        days |= set(pd.to_datetime(df_paradas_iv["dt_end"].dropna()).dt.normalize().tolist())

    if not days:
        return 0

    total_prod = 0
    dtv = pd.to_datetime(dt_verif, errors="coerce") if dt_verif is not None else pd.Series([], dtype="datetime64[ns]")

    for d in sorted(days):
        day_start = pd.Timestamp(d.strftime("%Y-%m-%d 00:00:00"))
        day_end = pd.Timestamp(d.strftime("%Y-%m-%d 23:59:59.999"))

        if dtv is not None and not dtv.dropna().empty:
            m_v = (dtv >= day_start) & (dtv <= day_end)
            df_v = df_verif.loc[m_v].copy()
            dt_v = dtv.loc[m_v].copy()
        else:
            df_v = df_verif.iloc[0:0].copy()
            dt_v = pd.Series([], dtype="datetime64[ns]")

        if df_paradas_iv is not None and not df_paradas_iv.empty and ("dt_start" in df_paradas_iv.columns) and (
                "dt_end" in df_paradas_iv.columns):
            m_p = (df_paradas_iv["dt_end"] >= day_start) & (df_paradas_iv["dt_start"] <= day_end)
            df_p = df_paradas_iv.loc[m_p].copy()
        else:
            df_p = None

        buckets = _compute_hourly_buckets(
            df_verif=df_v,
            dt_verif=dt_v,
            cols_verif=cols,
            selected_day=pd.Timestamp(d),
            df_paradas_iv=df_p,
        )

        # ✅ MISMA regla que en "day": si no hubo circuitos en esa hora, NO cuenta prodSec
        day_prod = 0
        for b in buckets:
            try:
                cut = int(b.get("cut") or 0)
                prod = int(b.get("prodSec") or 0)
            except Exception:
                continue

            if cut <= 0 and prod > 0:
                b["prodSec"] = 0
                prod = 0

            day_prod += prod

        # ✅ Para que semana/mes cuadre con la suma de los días (H:MM),
        #     truncamos a minuto igual que el KPI diario
        day_prod = (day_prod // 60) * 60

        total_prod += int(day_prod)

    return int(total_prod)


def _compute_unknown_seconds_for_range(
    df_verif: pd.DataFrame,
    dt_verif: pd.Series,
    cols: THBCols,
    df_paradas_iv: Optional[pd.DataFrame],
) -> int:
    """
    Suma el tiempo NO REGISTRADO (unknownSec) para week/month.
    Regla: suma unknownSec que generan los buckets por hora, día por día.
    """
    if (df_verif is None or df_verif.empty) and (df_paradas_iv is None or df_paradas_iv.empty):
        return 0

    # Días con actividad (verif y/o paradas)
    days = set()

    dtv = pd.to_datetime(dt_verif, errors="coerce") if dt_verif is not None else pd.Series([], dtype="datetime64[ns]")
    if not dtv.dropna().empty:
        days |= set(dtv.dropna().dt.normalize().unique().tolist())

    if df_paradas_iv is not None and not df_paradas_iv.empty:
        for c in ("dt_start", "dt_end"):
            if c in df_paradas_iv.columns:
                tt = pd.to_datetime(df_paradas_iv[c], errors="coerce").dropna()
                if not tt.empty:
                    days |= set(tt.dt.normalize().unique().tolist())

    if not days:
        return 0

    total_unknown = 0

    for d in sorted(days):
        day0 = pd.Timestamp(d).normalize()
        day1 = day0 + pd.Timedelta(days=1)

        # verif del día
        if not dtv.dropna().empty:
            m_v = (dtv >= day0) & (dtv < day1)
            df_v = df_verif.loc[m_v].copy()
            dt_v = dtv.loc[m_v].copy()
        else:
            df_v = df_verif.iloc[0:0].copy()
            dt_v = pd.Series([], dtype="datetime64[ns]")

        # paradas del día
        if (
            df_paradas_iv is not None
            and not df_paradas_iv.empty
            and ("dt_start" in df_paradas_iv.columns)
            and ("dt_end" in df_paradas_iv.columns)
        ):
            m_p = (df_paradas_iv["dt_end"] >= day0) & (df_paradas_iv["dt_start"] < day1)
            df_p = df_paradas_iv.loc[m_p].copy()
        else:
            df_p = None

        buckets = _compute_hourly_buckets(
            df_verif=df_v,
            dt_verif=dt_v,
            cols_verif=cols,
            selected_day=day0,
            df_paradas_iv=df_p,
        )

        total_unknown += int(sum(int(b.get("unknownSec", 0) or 0) for b in buckets))

    # Igual que el resto: truncar a minuto
    total_unknown = (int(total_unknown) // 60) * 60
    return int(total_unknown)



# ============================================================
#  KPIs (base) - OJO: Luego en analyze_thb se SOBREESCRIBEN los 2 KPIs fijos
# ============================================================
def _compute_daily_kpis(
    df_verif: pd.DataFrame,
    dt_verif: pd.Series,
    cols: THBCols,
    selected_day: Optional[pd.Timestamp],
    df_paradas_iv: Optional[pd.DataFrame] = None,
    *,
    period: str = "day",
) -> Dict[str, Any]:

    piezas_buenas = 0
    piezas_totales = 0

    if cols.col_piezas_buenas and cols.col_piezas_buenas in df_verif.columns:
        piezas_buenas = int(pd.to_numeric(df_verif[cols.col_piezas_buenas], errors="coerce").fillna(0).sum())

    if cols.col_total_piezas and cols.col_total_piezas in df_verif.columns:
        tot = pd.to_numeric(df_verif[cols.col_total_piezas], errors="coerce").fillna(0).astype(int)
        if cols.col_piezas_buenas and cols.col_piezas_buenas in df_verif.columns:
            buenas = pd.to_numeric(df_verif[cols.col_piezas_buenas], errors="coerce").fillna(0).astype(int)
            piezas_totales = int((tot.where(tot > 0, buenas)).sum())
        else:
            piezas_totales = int(tot.sum())
    else:
        piezas_totales = int(piezas_buenas)

    piezas_rechazo = int(piezas_buenas - piezas_totales)

    metros_cortados = 0.0
    metros_solicitados = 0.0

    if cols.col_piezas_buenas and cols.col_res_mm and (cols.col_piezas_buenas in df_verif.columns) and (cols.col_res_mm in df_verif.columns):
        buenas = pd.to_numeric(df_verif[cols.col_piezas_buenas], errors="coerce").fillna(0)
        resmm = df_verif[cols.col_res_mm].apply(_num_es)
        metros_cortados = float(((buenas * resmm) / 1000.0).sum())

    if cols.col_total_piezas and cols.col_long_mm and (cols.col_total_piezas in df_verif.columns) and (cols.col_long_mm in df_verif.columns):
        tot = pd.to_numeric(df_verif[cols.col_total_piezas], errors="coerce").fillna(0)
        longmm = df_verif[cols.col_long_mm].apply(_num_es)
        metros_solicitados = float(((tot * longmm) / 1000.0).sum())

    metros_extras = float(metros_cortados - metros_solicitados)

    # =========================================================
    # ✅ NOMINAL (THB) por tipo: t(L)=a+bL
    #   - q = piezas buenas
    #   - L = resultado verificación (mm)
    #   - tipo sale de Terminal1/Terminal2 (si existen; si no, asume tipo 1)
    #
    # KPIs:
    #   - tnom_period_sec  = Σ(q_i * tnom_i)
    #   - tcnp_period_sec  = tnom_period_sec / Σ(q_i)
    #   - tcnp_period_len_mm = Σ(q_i*L_i)/Σ(q_i)
    # =========================================================
    tcnp_period_sec = 0.0
    tcnp_period_len_mm = 0.0
    tnom_period_sec = 0.0

    if cols.col_piezas_buenas and cols.col_res_mm and (cols.col_piezas_buenas in df_verif.columns) and (cols.col_res_mm in df_verif.columns):
        buenas_all = pd.to_numeric(df_verif[cols.col_piezas_buenas], errors="coerce").fillna(0.0).astype(float)
        resmm_all = df_verif[cols.col_res_mm].apply(_num_es).astype(float)

        q_total = float(buenas_all.sum())
        sum_q_mm = float((buenas_all * resmm_all).sum())

        if q_total > 0 and sum_q_mm > 0:
            tcnp_period_len_mm = sum_q_mm / q_total

            # tipos por fila (si no hay terminales, asume tipo 1)
            if (
                getattr(cols, "col_terminal1", None) and getattr(cols, "col_terminal2", None)
                and (cols.col_terminal1 in df_verif.columns) and (cols.col_terminal2 in df_verif.columns)
            ):
                tipos = df_verif.apply(
                    lambda r: _thb_tipo_from_terminals(r.get(cols.col_terminal1), r.get(cols.col_terminal2)),
                    axis=1
                )
            else:
                tipos = pd.Series([1] * len(df_verif), index=df_verif.index)

            # tiempo nominal por fila (modelo "calc"; si quieres comparar v=8, cambia model="v8")
            t_nom = [
                _thb_nominal_sec(float(L), int(tp), model="calc")
                for L, tp in zip(resmm_all.tolist(), tipos.tolist())
            ]
            t_nom = pd.Series(t_nom, index=df_verif.index, dtype="float64")

            tnom_period_sec = float((buenas_all * t_nom).sum())
            tcnp_period_sec = float(tnom_period_sec / q_total) if q_total > 0 else 0.0

    # muerto/105 (esto quedará como “base”; luego lo sobreescribimos con la versión IGNORA-HORA)
    total_dead_sec = 0
    meal_105_sec = 0
    if df_paradas_iv is not None and not df_paradas_iv.empty:
        d = df_paradas_iv.copy()
        if "codigo" in d.columns:
            codes = d["codigo"].apply(_code_to_str)
        else:
            codes = pd.Series([""] * len(d), index=d.index)

        if "dur_sec" in d.columns:
            dur = pd.to_numeric(d["dur_sec"], errors="coerce").fillna(0.0).astype(float)
        else:
            dur = (pd.to_datetime(d["dt_end"], errors="coerce") - pd.to_datetime(d["dt_start"], errors="coerce")).dt.total_seconds()
            dur = pd.to_numeric(dur, errors="coerce").fillna(0.0).clip(lower=0.0)

        total_dead_sec = int(dur.sum())
        meal_105_sec = int(dur.where(codes.eq("105"), 0.0).sum())

    other_dead_sec = max(0, total_dead_sec - meal_105_sec)

    # tiempo efectivo (base; luego sobreescribimos con IGNORA-HORA)
    prod_sec = 0
    buckets: List[Dict[str, Any]] = []

    if period == "day":
        if selected_day is not None:
            buckets = _compute_hourly_buckets(
                df_verif=df_verif,
                dt_verif=dt_verif,
                cols_verif=cols,
                selected_day=selected_day,
                df_paradas_iv=df_paradas_iv,
            )
            prod_sec = int(sum(b.get("prodSec", 0) for b in buckets))

            # ✅ Para DAY: "Tiempo nominal" = suma de Tnom por hora (tarjetas)
            # (para que cuadre EXACTO con lo que el usuario ve)
            tnom_from_buckets = float(sum(float(b.get("tcnpTotalSec", 0) or 0) for b in buckets)) if buckets else 0.0
            if tnom_from_buckets > 0:
                tnom_period_sec = tnom_from_buckets
                if piezas_buenas > 0:
                    tcnp_period_sec = float(tnom_period_sec) / float(piezas_buenas)
                else:
                    tcnp_period_sec = 0.0
    else:
        prod_sec = _compute_prod_seconds_for_range(
            df_verif=df_verif,
            dt_verif=dt_verif,
            cols=cols,
            df_paradas_iv=df_paradas_iv,
        )
        buckets = []

        # (Para week/month ya queda usando df_verif completo del rango:
        #  tnom_period_sec y tcnp_period_sec ya están calculados arriba)

    return {
        "circuitos_cortados": piezas_buenas,
        "circuitos_planeados": piezas_totales,
        "circuitos_no_conformes": piezas_rechazo,

        "metros_cortados": metros_cortados,
        "metros_planeados": metros_solicitados,
        "metros_extras": metros_extras,

        "otro_tiempo_sec": other_dead_sec,
        "otro_tiempo_hhmm": _seconds_to_hhmm(float(other_dead_sec)),

        "parada_105_sec": meal_105_sec,
        "parada_105_hhmm": _seconds_to_hhmm(float(meal_105_sec)),

        "dead_total_sec": total_dead_sec,
        "dead_total_hhmm": _seconds_to_hhmm(float(total_dead_sec)),

        "tiempo_efectivo_sec": prod_sec,
        "tiempo_efectivo_hhmm": _format_hmm_from_seconds(prod_sec),

        # ✅ KPIs (THB): nominal ponderado del periodo
        "tnom_total_sec": float(tnom_period_sec),
        "tnom_total_hhmm": _seconds_to_hhmm(float(tnom_period_sec)),
        "tcnp_sec": float(tcnp_period_sec),          # seg/pieza (promedio ponderado)
        "tcnp_len_mm": float(tcnp_period_len_mm),    # mm/pieza (promedio ponderado)

        "hourly_buckets": buckets,
    }



# ============================================================
#  ✅ KPI FIX 1: Otro Tiempo ignora hora inicio/fin (usa 1ra y última hora REAL del día)
# ============================================================
def _compute_otro_tiempo_ignore_time(
    *,
    period: str,
    r0: pd.Timestamp,
    r1: pd.Timestamp,
    df_period_all: pd.DataFrame,
    dt_period_all: pd.Series,
    cols_nombre: Optional[str],
    operator: str,
    df_paradas_iv_all: Optional[pd.DataFrame],
) -> Tuple[int, int]:
    if df_paradas_iv_all is None or df_paradas_iv_all.empty or r0 is None or r1 is None:
        return 0, 0

    op = (operator or "").strip()
    is_general = (op == "" or op.lower() == "general")

    # timeline de verificación para actividad (por operaria si aplica)
    if (not is_general) and cols_nombre and (cols_nombre in df_period_all.columns):
        m_op = df_period_all[cols_nombre].astype(str).str.strip().eq(op)
        dt_act_all = pd.to_datetime(dt_period_all.loc[m_op], errors="coerce")
    else:
        dt_act_all = pd.to_datetime(dt_period_all, errors="coerce")

    total_dead = 0
    total_105 = 0

    day = pd.Timestamp(r0).normalize()
    end = pd.Timestamp(r1).normalize()

    while day < end:
        day0 = day
        day1 = day0 + pd.Timedelta(days=1)

        p = df_paradas_iv_all.copy()
        p["dt_start"] = pd.to_datetime(p["dt_start"], errors="coerce")
        p["dt_end"] = pd.to_datetime(p["dt_end"], errors="coerce")
        p = p.dropna(subset=["dt_start", "dt_end"])
        p = p.loc[(p["dt_end"] >= day0) & (p["dt_start"] < day1)].copy()

        v = dt_act_all.dropna()
        v = v[(v >= day0) & (v < day1)]

        times: List[pd.Timestamp] = []
        if not v.empty:
            times += v.tolist()
        if not p.empty:
            times += p["dt_start"].tolist()
            times += p["dt_end"].tolist()

        if times:
            w0 = min(times)
            w1 = max(times)

            if w0 < day0:
                w0 = day0
            if w1 > day1:
                w1 = day1

            if w1 > w0 and not p.empty:
                total_dead += _sum_overlap_seconds(p, w0, w1, code=None)
                total_105 += _sum_overlap_seconds(p, w0, w1, code="105")

        day = day1

    other = max(0, int(total_dead) - int(total_105))
    return other, int(total_105)


# ============================================================
#  ✅ KPI FIX 2: Tiempo Efectivo ignora hora inicio/fin
# ============================================================
def _compute_tiempo_efectivo_ignore_time(
    *,
    df_verif_all: pd.DataFrame,
    dt_verif_all: pd.Series,
    cols: THBCols,
    selected_day: Optional[pd.Timestamp],
    df_paradas_iv_all: Optional[pd.DataFrame],
    period: str,
) -> int:
    if period == "day":
        if selected_day is None:
            return 0
        buckets_all = _compute_hourly_buckets(
            df_verif=df_verif_all,
            dt_verif=dt_verif_all,
            cols_verif=cols,
            selected_day=selected_day,
            df_paradas_iv=df_paradas_iv_all,
        )
        return int(sum(b.get("prodSec", 0) for b in buckets_all))

    return int(_compute_prod_seconds_for_range(
        df_verif=df_verif_all,
        dt_verif=dt_verif_all,
        cols=cols,
        df_paradas_iv=df_paradas_iv_all,
    ))


# ============================================================
#  Pareto Paradas
# ============================================================
def _pareto_paradas(df_paradas_iv: pd.DataFrame, top_n: int = 25) -> Dict[str, Any]:
    empty = {
        "labels": [], "descs": [], "dur_sec": [], "dur_hhmm": [],
        "cum_pct": [], "total_sec": 0,
        "total_105_sec": 0, "total_other_sec": 0
    }

    if df_paradas_iv is None or df_paradas_iv.empty:
        return empty
    if "codigo" not in df_paradas_iv.columns or "dur_sec" not in df_paradas_iv.columns:
        return empty

    d = df_paradas_iv.copy()
    d["codigo"] = d["codigo"].apply(_code_to_str).astype(str).str.strip()
    d["dur_sec"] = pd.to_numeric(d["dur_sec"], errors="coerce").fillna(0.0).astype(float)
    d = d[(d["codigo"] != "") & (d["dur_sec"] > 0)].copy()
    if d.empty:
        return empty

    # 1) Suma REAL por código (segundos)
    raw_by_code = (
        d.groupby("codigo", dropna=False)["dur_sec"]
        .sum()
        .sort_values(ascending=False)
    )

    # 2) ✅ CUANTIZA A MINUTOS POR CÓDIGO (para que HH:MM sume exacto)
    #    - redondeo al minuto más cercano
    by_code_min = (raw_by_code / 60.0).round().astype(int)
    by_code_sec = (by_code_min * 60).astype(int)
    by_code_sec = by_code_sec[by_code_sec > 0]  # elimina filas 00:00

    if by_code_sec.empty:
        return empty

    # Totales (también cuantizados, por consistencia total)
    total_sec = int(by_code_sec.sum())
    total_105_sec = int(by_code_sec.get("105", 0))
    total_other_sec = max(0, total_sec - total_105_sec)

    # Top N (si recortas, esto ya NO rompe suma del KPI porque KPI usa los totales)
    g = by_code_sec.sort_values(ascending=False).head(top_n)

    # % acumulado sobre el TOTAL REAL cuantizado (no solo top_n)
    cum_pct = (g.cumsum() / float(total_sec) * 100.0).round(1).tolist() if total_sec > 0 else [0.0] * len(g)

    labels = [str(x) for x in g.index.tolist()]
    dur_sec = [int(x) for x in g.tolist()]
    dur_hhmm = [_seconds_to_hhmm(float(x)) for x in dur_sec]

    descs: List[str] = []
    for code in labels:
        descs.append(_desc_from_dict(code) or "")

    return {
        "labels": labels,
        "descs": descs,
        "dur_sec": dur_sec,
        "dur_hhmm": dur_hhmm,
        "cum_pct": cum_pct,
        "total_sec": int(total_sec),
        "total_105_sec": int(total_105_sec),
        "total_other_sec": int(total_other_sec),
    }




def build_pareto_paradas_thb(df_paradas: pd.DataFrame) -> dict:
    if df_paradas is None or df_paradas.empty:
        return {"labels": [], "descs": [], "dur_sec": [], "cum_pct": []}

    code_col = None
    for c in df_paradas.columns:
        if str(c).strip().lower() == "código":
            code_col = c
            break
    if code_col is None:
        for c in df_paradas.columns:
            if "codigo" in str(c).strip().lower():
                code_col = c
                break

    desc_col = None
    for c in df_paradas.columns:
        if "descrip" in str(c).strip().lower():
            desc_col = c
            break

    dur_col = None
    for c in df_paradas.columns:
        if "dur" in str(c).strip().lower():
            dur_col = c
            break

    if code_col is None or dur_col is None:
        return {"labels": [], "descs": [], "dur_sec": [], "cum_pct": []}

    d = df_paradas.copy()

    codes = pd.to_numeric(d[code_col], errors="coerce")
    d = d[codes.notna()].copy()
    d["_code"] = codes.loc[d.index].astype(int)

    d["_dur_sec"] = pd.to_numeric(d[dur_col], errors="coerce").fillna(0.0).astype(float)
    d = d[d["_dur_sec"] > 0].copy()

    if d.empty:
        return {"labels": [], "descs": [], "dur_sec": [], "cum_pct": []}

    if desc_col and desc_col in d.columns:
        d["_desc"] = d[desc_col].astype(str).str.strip()
        d.loc[d["_desc"].str.lower().isin(["nan", "none", ""]), "_desc"] = ""
    else:
        d["_desc"] = ""

    g = d.groupby("_code", as_index=False)["_dur_sec"].sum().sort_values("_dur_sec", ascending=False)

    descs = []
    for code in g["_code"].tolist():
        sub = d.loc[d["_code"] == code, "_desc"]
        sub = sub[sub != ""]
        descs.append("" if sub.empty else sub.value_counts().index[0])

    dur = g["_dur_sec"].to_numpy(dtype=float)
    total = float(dur.sum()) if dur.size else 0.0
    cum = (np.cumsum(dur) / total * 100.0) if total > 0 else np.zeros_like(dur)

    return {
        "labels": [str(int(x)) for x in g["_code"].tolist()],
        "descs": descs,
        "dur_sec": [float(x) for x in dur.tolist()],
        "cum_pct": [float(x) for x in cum.tolist()],
    }


# ============================================================
#  Top 10 Arnés
# ============================================================
def _top_arnes(df_verif: pd.DataFrame, cols: THBCols, top_n: int = 10) -> Dict[str, Any]:
    if not cols.col_arnes or cols.col_arnes not in df_verif.columns:
        return {"labels": [], "values": []}
    if not cols.col_piezas_buenas or cols.col_piezas_buenas not in df_verif.columns:
        return {"labels": [], "values": []}

    s_arnes = df_verif[cols.col_arnes].astype(str).str.strip()
    s_buenas = pd.to_numeric(df_verif[cols.col_piezas_buenas], errors="coerce").fillna(0)

    g = (
        pd.DataFrame({"arnes": s_arnes, "buenas": s_buenas})
        .query("arnes != '' and arnes != 'nan'")
        .groupby("arnes")["buenas"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    return {"labels": g.index.tolist(), "values": [int(x) for x in g.tolist()]}

def _thb_segment_counts_and_nominal(
    df_seg: pd.DataFrame,
    cols: THBCols,
) -> Dict[str, float]:
    out = {"cut": 0, "planned": 0, "tnom_sec": 0.0}

    if df_seg is None or df_seg.empty:
        return out

    if cols.col_piezas_buenas and cols.col_piezas_buenas in df_seg.columns:
        buenas = pd.to_numeric(df_seg[cols.col_piezas_buenas], errors="coerce").fillna(0.0).astype(float)
        out["cut"] = int(buenas.sum())
    else:
        buenas = pd.Series([0.0] * len(df_seg), index=df_seg.index, dtype="float64")

    if cols.col_total_piezas and cols.col_total_piezas in df_seg.columns:
        tot = pd.to_numeric(df_seg[cols.col_total_piezas], errors="coerce").fillna(0.0).astype(float)
        out["planned"] = int((tot.where(tot > 0, buenas)).sum())
    else:
        out["planned"] = int(out["cut"])

    if cols.col_res_mm and cols.col_res_mm in df_seg.columns:
        resmm = df_seg[cols.col_res_mm].apply(_num_es).astype(float)

        if (
            getattr(cols, "col_terminal1", None) and getattr(cols, "col_terminal2", None)
            and (cols.col_terminal1 in df_seg.columns) and (cols.col_terminal2 in df_seg.columns)
        ):
            tipos = df_seg.apply(
                lambda r: _thb_tipo_from_terminals(r.get(cols.col_terminal1), r.get(cols.col_terminal2)),
                axis=1
            )
        else:
            tipos = pd.Series([1] * len(df_seg), index=df_seg.index)

        t_nom = [
            _thb_nominal_sec(float(L), int(tp), model="calc")
            for L, tp in zip(resmm.tolist(), tipos.tolist())
        ]
        t_nom = pd.Series(t_nom, index=df_seg.index, dtype="float64")

        out["tnom_sec"] = float((buenas * t_nom).sum())

    return out


def _slice_paradas_intervals(
    df_paradas_iv: Optional[pd.DataFrame],
    seg0: pd.Timestamp,
    seg1: pd.Timestamp,
) -> pd.DataFrame:
    if df_paradas_iv is None or df_paradas_iv.empty:
        return pd.DataFrame(columns=["dt_start", "dt_end", "codigo", "desc", "dur_sec"])

    p = df_paradas_iv.copy()
    p["dt_start"] = pd.to_datetime(p["dt_start"], errors="coerce")
    p["dt_end"] = pd.to_datetime(p["dt_end"], errors="coerce")
    p = p.dropna(subset=["dt_start", "dt_end"])
    if p.empty:
        return p

    p = p.loc[(p["dt_end"] >= seg0) & (p["dt_start"] < seg1)].copy()
    if p.empty:
        return p

    return _clip_intervals_to_range(p, seg0, seg1)


def _clip01(x: float) -> float:
    try:
        n = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(n):
        return 0.0
    return float(max(0.0, min(1.0, n)))


def _build_thb_oee_chart(
    *,
    period: str,
    r0: Optional[pd.Timestamp],
    r1: Optional[pd.Timestamp],
    df_verif: pd.DataFrame,
    dt_verif: pd.Series,
    cols: THBCols,
    operator: str,
    df_paradas_iv: Optional[pd.DataFrame],
    df_verif_all_ops: Optional[pd.DataFrame] = None,
    dt_verif_all_ops: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    out = {
        "labels": [],
        "oee": [],
        "operacional": [],
        "disponibilidad": [],
        "calidad": [],
    }

    if r0 is None or r1 is None or df_verif is None or dt_verif is None:
        return out

    dtv = pd.to_datetime(dt_verif, errors="coerce")
    if dtv is None or dtv.empty:
        return out

    df_all_ops = df_verif_all_ops if isinstance(df_verif_all_ops, pd.DataFrame) else df_verif
    dt_all_ops = pd.to_datetime(dt_verif_all_ops, errors="coerce") if dt_verif_all_ops is not None else dtv

    def _push(label: str, cut: float, planned: float, tnom_sec: float, teff_sec: float, tpaid_sec: float):
        calidad = _clip01((planned / cut) if cut > 0 else 0.0)
        disponibilidad = _clip01((teff_sec / tpaid_sec) if tpaid_sec > 0 else 0.0)
        operacional = _clip01((tnom_sec / teff_sec) if teff_sec > 0 else 0.0)
        oee = _clip01(disponibilidad * operacional * calidad)

        out["labels"].append(str(label))
        out["oee"].append(round(oee * 100.0, 1))
        out["operacional"].append(round(operacional * 100.0, 1))
        out["disponibilidad"].append(round(disponibilidad * 100.0, 1))
        out["calidad"].append(round(calidad * 100.0, 1))

    if period == "day":
        day = pd.Timestamp(r0).normalize()
        p_day = _slice_paradas_intervals(df_paradas_iv, day, day + pd.Timedelta(days=1))

        buckets = _compute_hourly_buckets(
            df_verif=df_verif,
            dt_verif=dtv,
            cols_verif=cols,
            selected_day=day,
            df_paradas_iv=p_day,
        )

        for b in buckets:
            htxt = str(b.get("hour") or "").strip()
            if not htxt.isdigit():
                continue

            h = int(htxt)
            seg0 = pd.Timestamp(f"{day.strftime('%Y-%m-%d')} {h:02d}:00:00")
            seg1 = seg0 + pd.Timedelta(hours=1)

            m_seg = dtv.notna() & (dtv >= seg0) & (dtv < seg1)
            df_seg = df_verif.loc[m_seg].copy()

            base = _thb_segment_counts_and_nominal(df_seg, cols)

            cut = float(base["cut"])
            planned = float(base["planned"])
            tnom_sec = float(base["tnom_sec"])
            teff_sec = float(b.get("prodSec", 0) or 0)
            tpaid_sec = float(b.get("capacitySec", 3600) or 3600)

            if cut <= 0 and planned <= 0 and tnom_sec <= 0 and teff_sec <= 0 and float(b.get("deadSec", 0) or 0) <= 0:
                continue

            _push(
                label=f"{h:02d}",
                cut=cut,
                planned=planned,
                tnom_sec=tnom_sec,
                teff_sec=teff_sec,
                tpaid_sec=tpaid_sec,
            )

        return out

    DAY_ABBR = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

    if period == "week":
        day = pd.Timestamp(r0).normalize()

        while day < pd.Timestamp(r1):
            seg0 = day
            seg1 = min(day + pd.Timedelta(days=1), pd.Timestamp(r1))

            m_seg = dtv.notna() & (dtv >= seg0) & (dtv < seg1)
            df_seg = df_verif.loc[m_seg].copy()
            dt_seg = dtv.loc[m_seg].copy()
            p_seg = _slice_paradas_intervals(df_paradas_iv, seg0, seg1)

            if df_seg.empty and p_seg.empty:
                day = seg1
                continue

            k_seg = _compute_daily_kpis(
                df_verif=df_seg,
                dt_verif=dt_seg,
                cols=cols,
                selected_day=seg0,
                df_paradas_iv=p_seg,
                period="day",
            )

            m_all = dt_all_ops.notna() & (dt_all_ops >= seg0) & (dt_all_ops < seg1)
            df_seg_all = df_all_ops.loc[m_all].copy() if isinstance(df_all_ops, pd.DataFrame) else df_seg
            dt_seg_all = dt_all_ops.loc[m_all].copy() if dt_all_ops is not None else dt_seg

            tpaid_sec = float(_compute_horas_hombre_thb(
                period="day",
                r0=seg0,
                r1=seg1,
                df_verif=df_seg,
                dt_verif=dt_seg,
                cols=cols,
                operator=operator,
                df_paradas_iv=p_seg,
                df_verif_all_ops=df_seg_all,
                dt_verif_all_ops=dt_seg_all,
            ))

            _push(
                label=f"{DAY_ABBR[int(seg0.weekday())]} {seg0.strftime('%d')}",
                cut=float(k_seg.get("circuitos_cortados", 0) or 0),
                planned=float(k_seg.get("circuitos_planeados", 0) or 0),
                tnom_sec=float(k_seg.get("tnom_total_sec", 0) or 0),
                teff_sec=float(k_seg.get("tiempo_efectivo_sec", 0) or 0),
                tpaid_sec=tpaid_sec,
            )

            day = seg1

        return out

    if period == "month":
        seg0 = pd.Timestamp(r0).normalize()
        week_idx = 1

        while seg0 < pd.Timestamp(r1):
            seg1 = min(seg0 + pd.Timedelta(days=7), pd.Timestamp(r1))

            m_seg = dtv.notna() & (dtv >= seg0) & (dtv < seg1)
            df_seg = df_verif.loc[m_seg].copy()
            dt_seg = dtv.loc[m_seg].copy()
            p_seg = _slice_paradas_intervals(df_paradas_iv, seg0, seg1)

            if df_seg.empty and p_seg.empty:
                seg0 = seg1
                continue

            k_seg = _compute_daily_kpis(
                df_verif=df_seg,
                dt_verif=dt_seg,
                cols=cols,
                selected_day=None,
                df_paradas_iv=p_seg,
                period="week",
            )

            m_all = dt_all_ops.notna() & (dt_all_ops >= seg0) & (dt_all_ops < seg1)
            df_seg_all = df_all_ops.loc[m_all].copy() if isinstance(df_all_ops, pd.DataFrame) else df_seg
            dt_seg_all = dt_all_ops.loc[m_all].copy() if dt_all_ops is not None else dt_seg

            tpaid_sec = float(_compute_horas_hombre_thb(
                period="week",
                r0=seg0,
                r1=seg1,
                df_verif=df_seg,
                dt_verif=dt_seg,
                cols=cols,
                operator=operator,
                df_paradas_iv=p_seg,
                df_verif_all_ops=df_seg_all,
                dt_verif_all_ops=dt_seg_all,
            ))

            _push(
                label=f"Sem {week_idx}",
                cut=float(k_seg.get("circuitos_cortados", 0) or 0),
                planned=float(k_seg.get("circuitos_planeados", 0) or 0),
                tnom_sec=float(k_seg.get("tnom_total_sec", 0) or 0),
                teff_sec=float(k_seg.get("tiempo_efectivo_sec", 0) or 0),
                tpaid_sec=tpaid_sec,
            )

            week_idx += 1
            seg0 = seg1

        return out

    return out
# ============================================================
# ✅ analyze_thb (COMPLETA)
#  - todo lo demás respeta time_start/time_end
#  - PERO estos 2 KPIs NO cambian con la hora:
#      * Otro Tiempo de Ciclo (Muerto)
#      * Tiempo Efectivo (Día)
# ============================================================
def analyze_thb(
    df_verif: pd.DataFrame,
    *,
    period: str = "day",
    period_value: str = "",
    machine: str = "THB",
    operator: str = "",
    time_start: str = "",
    time_end: str = "",
    df_paradas: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:

    cols = _resolve_thb_cols(df_verif)
    if not cols.col_fecha:
        return {"status": "error", "message": "No se encontró columna 'Fecha' en Verificación de Medidas.", "machine": machine}

    pv = _normalize_period_value(period, period_value)
    r0, r1 = _period_bounds(period, pv)

    dt_all = _build_dt(df_verif, cols.col_fecha, cols.col_hora)

    if r0 is not None and r1 is not None:
        m_period = dt_all.notna() & (dt_all >= r0) & (dt_all < r1)
    else:
        m_period = dt_all.notna() & pd.Series([False] * len(dt_all), index=dt_all.index)

    df_period_all = df_verif.loc[m_period].copy()
    dt_period_all = dt_all.loc[m_period].copy()

    # --- filtra por operaria (esto SÍ debe afectar los KPIs fijos también) ---
    df_f = df_period_all.copy()
    dt_f = dt_period_all.copy()

    op_is_general = _is_general_operator(operator)
    if (not op_is_general) and cols.col_nombre and (cols.col_nombre in df_f.columns):
        sel = str(operator).strip()
        m_op = df_f[cols.col_nombre].astype(str).str.strip() == sel
        df_f = df_f.loc[m_op].copy()
        dt_f = dt_f.loc[m_op].copy()

    # ✅ copias SIN FILTRO HORA (para KPIs fijos)
    df_no_hour = df_f.copy()
    dt_no_hour = dt_f.copy()

    # --- aplica ventana horaria para el resto ---
    if time_start and time_end and dt_f is not None and (not dt_f.empty):
        m_time = _mask_time_window(dt_f, time_start, time_end)
        df_f = df_f.loc[m_time].copy()
        dt_f = dt_f.loc[m_time].copy()

    selected_day = None
    if period == "day" and r0 is not None:
        selected_day = pd.Timestamp(r0)

    # =========================================================
    # PARADAS -> base por rango + operaria
    #   all : SIN filtro hora
    #   eff : CON filtro hora (para cosas dependientes de ventana)
    # =========================================================
    df_paradas_iv_all = None
    df_paradas_iv_eff = None

    if df_paradas is not None and isinstance(df_paradas, pd.DataFrame) and not df_paradas.empty:
        base = _build_paradas_intervals(df_paradas)

        if r0 is not None and r1 is not None and base is not None and not base.empty:
            m_pr = (
                base["dt_end"].notna()
                & base["dt_start"].notna()
                & (base["dt_end"] >= r0)
                & (base["dt_start"] < r1)
            )
            base = base.loc[m_pr].copy()

        # ✅ recortar duraciones al rango real del periodo (evita inflar totales)
        if r0 is not None and r1 is not None and base is not None and not base.empty:
            base = _clip_intervals_to_range(base, r0, r1)

        base = _filter_paradas_by_operator_using_verif(
            df_paradas_iv=base,
            df_verif=df_period_all,
            dt_verif=dt_period_all,
            cols_verif=cols,
            operator=operator,
        )

        df_paradas_iv_all = base

        df_paradas_iv_eff = base
        if time_start and time_end and df_paradas_iv_eff is not None and not df_paradas_iv_eff.empty:
            m_pt = _mask_time_window(
                pd.to_datetime(df_paradas_iv_eff["dt_end"], errors="coerce"),
                time_start,
                time_end,
            )
            df_paradas_iv_eff = df_paradas_iv_eff.loc[m_pt].copy()

    # =========================================================
    # KPIs base (pueden depender de la hora) -> luego sobreescribimos tiempos
    # =========================================================
    kpis = _compute_daily_kpis(
        df_verif=df_f,
        dt_verif=dt_f,
        cols=cols,
        selected_day=selected_day,
        df_paradas_iv=df_paradas_iv_eff,
        period=period,
    )

    # =========================================================
    # ✅ AJUSTE: si una hora tiene 0 circuitos pero trae prodSec,
    #           ese prodSec NO debe contar en Tiempo Efectivo.
    # =========================================================
    removed_prod_zero_circuits = 0

    if period == "day":
        buckets = kpis.get("hourly_buckets")
        if isinstance(buckets, list) and buckets:
            for b in buckets:
                try:
                    cut = int(b.get("cut") or 0)
                    prod = int(b.get("prodSec") or 0)
                except Exception:
                    continue

                if cut <= 0 and prod > 0:
                    removed_prod_zero_circuits += prod
                    b["prodSec"] = 0

    # =========================================================
    # ✅ 2.4: FUENTE ÚNICA para TIEMPOS (IGNORA hora)
    # =========================================================
    use_full_time = (period in ("day", "week", "month"))

    paradas_time_src = df_paradas_iv_all if use_full_time else df_paradas_iv_eff
    verif_time_src_df = df_no_hour if use_full_time else df_f
    verif_time_src_dt = dt_no_hour if use_full_time else dt_f

    # 1) Pareto desde la MISMA fuente que define "Otro tiempo"
    pareto = _pareto_paradas(paradas_time_src) if paradas_time_src is not None else {
        "labels": [], "descs": [], "dur_sec": [], "dur_hhmm": [], "cum_pct": [],
        "total_sec": 0, "total_105_sec": 0, "total_other_sec": 0
    }
    # ✅ Parada 106 (misma lógica de cuantización que Pareto)
    # ✅ Parada 104 (misma lógica de cuantización que Pareto)
    p104_sec = 0
    if paradas_time_src is not None and not paradas_time_src.empty and "codigo" in paradas_time_src.columns:
        tmp104 = paradas_time_src.copy()
        tmp104["codigo"] = tmp104["codigo"].apply(_code_to_str).astype(str).str.strip()

        if "dur_sec" in tmp104.columns:
            s104 = pd.to_numeric(
                tmp104.loc[tmp104["codigo"].eq("104"), "dur_sec"],
                errors="coerce"
            ).fillna(0.0).sum()
        else:
            s104 = 0.0

        # cuantiza EXACTO igual que Pareto: segundos -> minutos redondeados -> segundos
        m104 = int(round(float(s104) / 60.0))
        p104_sec = max(0, m104 * 60)

    kpis["parada_104_sec"] = int(p104_sec)
    kpis["parada_104_hhmm"] = _seconds_to_hhmm(float(p104_sec))

    # 2) KPI "Otro Tiempo" = suma exacta del Pareto (sin 105)
    # "Otro" = (total - 105) PERO sin 106
    # "Otro" = (total - 105) PERO sin 104
    other_from_pareto = int(pareto.get("total_other_sec", 0) or 0)
    kpis["otro_tiempo_sec"] = max(0, other_from_pareto - int(p104_sec or 0))
    kpis["otro_tiempo_hhmm"] = _seconds_to_hhmm(float(kpis["otro_tiempo_sec"]))

    kpis["otro_tiempo_hhmm"] = _seconds_to_hhmm(float(kpis["otro_tiempo_sec"]))

    kpis["parada_105_sec"] = int(pareto.get("total_105_sec", 0))
    kpis["parada_105_hhmm"] = _seconds_to_hhmm(float(kpis["parada_105_sec"]))

    kpis["dead_total_sec"] = int(pareto.get("total_sec", 0))
    kpis["dead_total_hhmm"] = _seconds_to_hhmm(float(kpis["dead_total_sec"]))

    # 3) Tiempo Efectivo y buckets
    if period == "day" and selected_day is not None:
        buckets_full = _compute_hourly_buckets(
            df_verif=verif_time_src_df,
            dt_verif=verif_time_src_dt,
            cols_verif=cols,
            selected_day=selected_day,
            df_paradas_iv=paradas_time_src,
        )

        # ✅ AJUSTE REAL (sobre buckets_full que son los que se muestran):
        if isinstance(buckets_full, list) and buckets_full:
            for b in buckets_full:
                try:
                    cut = int(b.get("cut") or 0)
                    prod = int(b.get("prodSec") or 0)
                except Exception:
                    continue
                if cut <= 0 and prod > 0:
                    b["prodSec"] = 0

        # ✅✅ NUEVO: agrega detalle de causas por hora (para "Otro")
        _attach_other_causes_to_buckets_thb(buckets_full, paradas_time_src, selected_day)

        kpis["hourly_buckets"] = buckets_full

        # ✅ NO REGISTRADO (día) = suma unknownSec de las tarjetas (buckets)
        no_reg_sec = int(sum(int(b.get("unknownSec", 0) or 0) for b in buckets_full))
        no_reg_sec = (no_reg_sec // 60) * 60
        kpis["no_registrado_sec"] = int(no_reg_sec)
        kpis["no_registrado_hhmm"] = _seconds_to_hhmm(float(no_reg_sec))

        prod_full = int(sum(int(b.get("prodSec", 0)) for b in buckets_full))
        kpis["tiempo_efectivo_sec"] = prod_full
        kpis["tiempo_efectivo_hhmm"] = _format_hmm_from_seconds(prod_full)

    else:
        prod_full = _compute_tiempo_efectivo_ignore_time(
            df_verif_all=df_no_hour,
            dt_verif_all=dt_no_hour,
            cols=cols,
            selected_day=selected_day,
            df_paradas_iv_all=df_paradas_iv_all,
            period=period,
        )

        prod_full_adj = max(0, int(prod_full) - int(removed_prod_zero_circuits))

        kpis["tiempo_efectivo_sec"] = int(prod_full_adj)
        kpis["tiempo_efectivo_hhmm"] = _format_hmm_from_seconds(prod_full_adj)

        # ✅ NO REGISTRADO (week/month) = usa tu función _compute_unknown_seconds_for_range
        no_reg_sec = _compute_unknown_seconds_for_range(
            df_verif=df_no_hour,
            dt_verif=dt_no_hour,
            cols=cols,
            df_paradas_iv=df_paradas_iv_all,
        )
        kpis["no_registrado_sec"] = int(no_reg_sec)
        kpis["no_registrado_hhmm"] = _seconds_to_hhmm(float(no_reg_sec))

        kpis["hourly_buckets"] = []

    # =========================================================
    # ✅ KPI NUEVO: Horas/Hombre (ignora ventana horaria)
    # =========================================================
    hh_sec = _compute_horas_hombre_thb(
period=period,
        r0=r0,
        r1=r1,
        df_verif=df_no_hour,          # <- sin filtro hora
        dt_verif=dt_no_hour,          # <- sin filtro hora
        cols=cols,
        operator=operator,            # <- respeta filtro operaria
        df_paradas_iv=df_paradas_iv_all,
            df_verif_all_ops=df_period_all,
        dt_verif_all_ops=dt_period_all,
    )
    kpis["horas_hombre_sec"] = int(hh_sec)
    kpis["horas_hombre_hhmm"] = _format_hmm_from_seconds(int(hh_sec))

    # Top arnés (queda como estaba: depende de df_f con ventana)
    top_arnes = _top_arnes(df_f, cols)

    dt_min = dt_f.min() if dt_f is not None and not dt_f.dropna().empty else None
    dt_max = dt_f.max() if dt_f is not None and not dt_f.dropna().empty else None

    lotes_acumulados = _build_thb_lotes_acumulados(
        df_verif=df_no_hour,
        dt_verif=dt_no_hour,
        cols=cols,
    )

    # =========================================================
    # ✅ NO REGISTRADO (LÓGICA FINAL) — 105 y 106 NO se mezclan con "Otro"
    #   - "Otro" = paradas !=105 y !=106
    #   - "No Registrado" = Horas/Hombre - (Efectivo + Otro)
    #   - Si se pasa: NoRegistrado=0 y el excedente se mueve a 105 (opcional)
    # =========================================================
    hh_sec = int(kpis.get("horas_hombre_sec", 0) or 0)
    eff_sec = int(kpis.get("tiempo_efectivo_sec", 0) or 0)

    p105_sec = int(kpis.get("parada_105_sec", 0) or 0)
    p106_sec = int(kpis.get("parada_106_sec", 0) or 0)

    # 👉 OJO: pareto["total_other_sec"] = (total - 105) PERO incluye 106.
    # Entonces: otro_sin_105_106 = total_other_sec - 106
    other_sec = int(kpis.get("otro_tiempo_sec", 0) or 0)  # ✅ ya viene sin 106


    # trabajar en minutos para cuadrar exacto
    hh_m = max(0, hh_sec // 60)
    eff_m = max(0, eff_sec // 60)
    other_m = max(0, other_sec // 60)
    p105_m = max(0, p105_sec // 60)
    p106_m = max(0, p106_sec // 60)

    used_m = eff_m + other_m

    if used_m <= hh_m:
        nr_m = hh_m - used_m
    else:
        overflow_m = used_m - hh_m
        take_m = min(other_m, overflow_m)
        other_m -= take_m

        # ✅ CAMBIO: el excedente ya NO se va a 105
        # ✅ ahora se registra como "No registrado"
        nr_m = overflow_m

    # escribir de vuelta
    kpis["otro_tiempo_sec"] = int(other_m * 60)
    kpis["otro_tiempo_hhmm"] = _seconds_to_hhmm(float(kpis["otro_tiempo_sec"]))

    kpis["parada_105_sec"] = int(p105_m * 60)
    kpis["parada_105_hhmm"] = _seconds_to_hhmm(float(kpis["parada_105_sec"]))

    kpis["parada_106_sec"] = int(p106_m * 60)
    kpis["parada_106_hhmm"] = _seconds_to_hhmm(float(kpis["parada_106_sec"]))

    kpis["no_registrado_sec"] = int(nr_m * 60)
    kpis["no_registrado_hhmm"] = _seconds_to_hhmm(float(kpis["no_registrado_sec"]))

    # =========================================================
    # ✅ PORCENTAJES vs Horas/Hombre (100%)
    #   - 105 NO cuenta
    #   - %Efectivo = TiempoEfectivo / HorasHombre
    #   - %Otro+NR  = (Otro + NoRegistrado) / HorasHombre
    #     (como 105 no cuenta, esto debe quedar como el "resto" del 100%)
    # =========================================================
    hh_m = int(kpis.get("horas_hombre_sec", 0) or 0) // 60
    eff_m = int(kpis.get("tiempo_efectivo_sec", 0) or 0) // 60

    if hh_m <= 0:
        pct_eff = 0.0
        pct_other_nr = 0.0
    else:
        pct_eff = round((eff_m / hh_m) * 100.0, 1)
        # fuerza el resto para que siempre sume 100% (evita problemas de redondeo)
        pct_other_nr = round(100.0 - pct_eff, 1)

    # clamps por seguridad
    pct_eff = float(max(0.0, min(100.0, pct_eff)))
    pct_other_nr = float(max(0.0, min(100.0, pct_other_nr)))

    kpis["pct_efectivo"] = pct_eff
    kpis["pct_otro_no_reg"] = pct_other_nr

    # =========================================================
    # ✅ KPI: Índice Operacional (IO) = Tnom / Tefectivo
    #   - Tnom: tiempo nominal total del periodo (seg)
    #   - Teff: tiempo efectivo (prod) del periodo (seg)
    #   - En número (no %)
    # =========================================================
    tnom_sec = float(kpis.get("tnom_total_sec", 0) or 0)
    teff_sec = float(kpis.get("tiempo_efectivo_sec", 0) or 0)

    io = (tnom_sec / teff_sec) if teff_sec > 0 else 0.0
    kpis["indice_operacional"] = float(io)

    # ✅ KPI: Velocidad Nominal (VN) = 3600 / TCNP
    tcnp_sec_period = float(kpis.get("tcnp_sec", 0.0) or 0.0)
    vn_uph = (3600.0 / tcnp_sec_period) if tcnp_sec_period > 0 else 0.0
    kpis["velocidad_nominal_uph"] = float(vn_uph)

    # =========================
    # ✅ OEE breakdown (Rendimiento / Disponibilidad / Calidad)
    # =========================
    good = float(kpis.get("circuitos_cortados", 0) or 0)
    total = float(kpis.get("circuitos_planeados", 0) or 0)

    teff = float(kpis.get("tiempo_efectivo_sec", 0) or 0)          # seg
    tpaid = float(kpis.get("horas_hombre_sec", 0) or 0)            # seg
    tnom = float(kpis.get("tnom_total_sec", 0) or 0)               # seg

    calidad = (total / good) if good > 0 else 0.0                 # Q
    disponibilidad = (teff / tpaid) if tpaid > 0 else 0.0          # A
    rendimiento = (tnom / teff) if teff > 0 else 0.0               # P  (tu IO)

    oee = disponibilidad * rendimiento * calidad

    kpis["oee"] = float(oee)
    kpis["oee_calidad"] = float(calidad)
    kpis["oee_disponibilidad"] = float(disponibilidad)
    kpis["oee_rendimiento"] = float(rendimiento)

    # =========================
    # ✅ Cumplimiento del plan
    # Misma idea del Excel:
    # PP = OEE esperado * VN * TP
    # Cumplimiento = Producción Total / PP
    # =========================
    OEE_ESPERADO = 0.70  # si luego cambias ese 70% en el Excel, cambia también aquí

    tp_h = float(kpis.get("horas_hombre_sec", 0) or 0) / 3600.0
    vn_uph = float(kpis.get("velocidad_nominal_uph", 0.0) or 0.0)

    produccion_planeada_calc = OEE_ESPERADO * vn_uph * tp_h
    cumplimiento_plan = (good / produccion_planeada_calc) if produccion_planeada_calc > 0 else 0.0

    kpis["produccion_planeada_calc"] = float(produccion_planeada_calc)
    kpis["cumplimiento_plan"] = float(cumplimiento_plan)

    ui = {
        "Producción Total": f"{kpis['circuitos_cortados']:,}".replace(",", "."),
        "Producción Buena": f"{kpis['circuitos_planeados']:,}".replace(",", "."),
        "Producción con Defectos": f"{kpis['circuitos_no_conformes']:,}".replace(",", "."),

        "Tiempo Perdido": (
            f"{kpis['otro_tiempo_hhmm']}||"
            f"105: {kpis['parada_105_hhmm']} | "
            f"No registrado: {kpis['no_registrado_hhmm']} | "
            f"104: {kpis.get('parada_104_hhmm', '00:00')} | "
            f"{kpis.get('pct_otro_no_reg', 0.0):.1f}%"
        ),

        "Tiempo Trabajado": f"{kpis['tiempo_efectivo_hhmm']} ({kpis.get('pct_efectivo', 0.0):.1f}%)",
        "Tiempo Pagado": kpis["horas_hombre_hhmm"],

        "Cumplimiento del plan": f"{(kpis.get('cumplimiento_plan', 0.0) * 100):.1f}%",

        "Metros Cortados":  f"{kpis['metros_cortados']:,.1f} m".replace(",", "X").replace(".", ",").replace("X", "."),
        "Metros Planeados": f"{kpis['metros_planeados']:,.1f} m".replace(",", "X").replace(".", ",").replace("X", "."),
        "Metros Extras":    f"{kpis['metros_extras']:,.1f} m".replace(",", "X").replace(".", ",").replace("X", "."),
        "Velocidad Nominal": f"{int(round(kpis.get('velocidad_nominal_uph', 0.0))):,} unid/h".replace(",", "."),

        "Tiempo De Corte": kpis.get("tnom_total_hhmm", "00:00"),
        #"Índice Operacional (IO)": f"{kpis.get('indice_operacional', 0.0):.3f}",
        "Tiempo de Ciclo": f"{kpis.get('tcnp_sec', 0.0):.4f} s/pz",

        "OEE": (
            f"{(kpis.get('oee', 0.0) * 100):.1f}%||"
            f"Indice de Eficiencia Operacional: {(kpis.get('oee_rendimiento', 0.0) * 100):.1f}% | "
            f"Indice de Disponibilidad: {(kpis.get('oee_disponibilidad', 0.0) * 100):.1f}% | "
            f"Indice de Calidad: {(kpis.get('oee_calidad', 0.0) * 100):.1f}%"
        ),


    }

    chart_df_verif = df_f if period == "day" else df_no_hour
    chart_dt_verif = dt_f if period == "day" else dt_no_hour
    chart_df_paradas = df_paradas_iv_eff if period == "day" else df_paradas_iv_all

    oee_chart = _build_thb_oee_chart(
        period=period,
        r0=r0,
        r1=r1,
        df_verif=chart_df_verif,
        dt_verif=chart_dt_verif,
        cols=cols,
        operator=operator,
        df_paradas_iv=chart_df_paradas,
        df_verif_all_ops=df_period_all,
        dt_verif_all_ops=dt_period_all,
    )

    return {
        "status": "ok",
        "analysis": ANALYSIS_KEY,
        "title": ANALYSIS_TITLE,

        "machine": machine,
        "period": period,
        "period_value": pv,
        "operator": operator,
        "time_start": time_start,
        "time_end": time_end,

        "rows_total": int(len(df_f)),
        "rows_with_datetime_in_period": int(
            pd.to_datetime(dt_f, errors="coerce").notna().sum()) if dt_f is not None else 0,

        "dt_min": dt_min.isoformat(sep=" ") if dt_min is not None and not pd.isna(dt_min) else None,
        "dt_max": dt_max.isoformat(sep=" ") if dt_max is not None and not pd.isna(dt_max) else None,

        "kpis": kpis,
        "kpis_ui": ui,
        "pareto": pareto,
        "top_arnes": top_arnes,
        "lotes_acumulados": lotes_acumulados,
        "oee_chart": oee_chart,
    }



def _thb_net_shift_seconds_for_day(day: pd.Timestamp) -> int:
    """
    Retorna segundos netos por operaria según tu calendario THB.
    Lun-Jue: 07:00-17:00 menos (30 almuerzo + 15 desayuno + 7 pausas) = 9h08m
    Vie:     07:00-14:15 menos (15 desayuno + 7 pausas) = 6h53m
    """
    d = pd.Timestamp(day).normalize()
    wd = int(d.weekday())  # 0=lun ... 4=vie ... 6=dom

    if wd in (0, 1, 2, 3):
        return 10 * 3600 - (30 + 15 + 7) * 60  # 32880 sec = 9:08
    if wd == 4:
        return (7 * 3600 + 15 * 60) - (15 + 7) * 60  # 24780 sec = 6:53
    return 0  # sábado/domingo (sin jornada definida)


def _compute_horas_hombre_thb(
    *,
    period: str,
    r0: Optional[pd.Timestamp],
    r1: Optional[pd.Timestamp],
    df_verif: pd.DataFrame,
    dt_verif: pd.Series,
    cols: THBCols,
    operator: str,
    df_paradas_iv: Optional[pd.DataFrame],
    df_verif_all_ops: Optional[pd.DataFrame] = None,
    dt_verif_all_ops: Optional[pd.Series] = None,
) -> int:
    """
    Tiempo Pagado (Horas/Hombre) THB.

    - Si operator == General: usa regla de turno por # tarjetas (9.25 / 10.25 / 11.25 ...).
    - Si operator != General:
        - Si en el día hubo >=2 operarias: paga por horas trabajadas de esa operaria (cards * 1h)
          y descuenta Parada 105 SOLO en ventanas fijas:
              08:30-08:45 (15 min) y 13:30-14:00 (30 min),
          pero solo si realmente hay intervalos 105 solapando esas ventanas.
        - Si en el día hubo solo 1 operaria: se comporta igual que General.
    """

    if (df_verif is None or df_verif.empty) and (df_paradas_iv is None or df_paradas_iv.empty):
        return 0

    # -------------------------
    # Helpers
    # -------------------------
    def _bucket_total_sec(b: Dict[str, Any]) -> int:
        try:
            prod = int(b.get("prodSec") or 0)
            dead = int(b.get("deadSec") or 0)
            unk  = int(b.get("unknownSec") or 0)
            meal = int(b.get("mealSec") or 0)
            return max(0, prod + dead + unk + meal)
        except Exception:
            return 0

    def _count_cards(buckets: List[Dict[str, Any]]) -> int:
        if not buckets:
            return 0
        n = 0
        for b in buckets:
            if _bucket_total_sec(b) > 0:
                n += 1
        return int(n)

    def _overlap(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> int:
        # segundos de solape entre [a0,a1) y [b0,b1)
        start = max(a0.value, b0.value)
        end = min(a1.value, b1.value)
        return int(max(0, (end - start) / 1e9))

    def _sum_overlap_105_in_window(df_p: Optional[pd.DataFrame], w0: pd.Timestamp, w1: pd.Timestamp) -> int:
        # suma de solape con código 105 dentro de la ventana [w0,w1)
        if df_p is None or df_p.empty:
            return 0
        if ("dt_start" not in df_p.columns) or ("dt_end" not in df_p.columns):
            return 0
        d = df_p.copy()
        if "codigo" not in d.columns:
            return 0
        d["codigo"] = d["codigo"].apply(_code_to_str).astype(str).str.strip()
        d = d.loc[d["codigo"].eq("105")].copy()
        if d.empty:
            return 0
        d["dt_start"] = pd.to_datetime(d["dt_start"], errors="coerce")
        d["dt_end"] = pd.to_datetime(d["dt_end"], errors="coerce")
        d = d.dropna(subset=["dt_start", "dt_end"])
        if d.empty:
            return 0

        total = 0
        for _, r in d.iterrows():
            s = r["dt_start"]
            e = r["dt_end"]
            if e <= w0 or s >= w1:
                continue
            total += _overlap(s, e, w0, w1)
        # no puede exceder la ventana
        return int(min(total, int((w1 - w0).total_seconds())))

    # -------------------------
    # Días con datos (verif + paradas)
    # -------------------------
    days = set()

    if dt_verif is not None and not dt_verif.dropna().empty:
        dd = pd.to_datetime(dt_verif.dropna(), errors="coerce").dropna()
        if not dd.empty:
            days |= set(dd.dt.normalize().tolist())

    if df_paradas_iv is not None and not df_paradas_iv.empty:
        for c in ("dt_start", "dt_end"):
            if c in df_paradas_iv.columns:
                tt = pd.to_datetime(df_paradas_iv[c], errors="coerce").dropna()
                if not tt.empty:
                    days |= set(tt.dt.normalize().tolist())

    if not days:
        return 0

    # timeline principal (posiblemente filtrada por operaria)
    dtv = pd.to_datetime(dt_verif, errors="coerce") if dt_verif is not None else pd.Series([], dtype="datetime64[ns]")

    # timeline "all ops" (sin filtrar operaria) para contar operarias del día
    dt_all_ops = pd.to_datetime(dt_verif_all_ops, errors="coerce") if dt_verif_all_ops is not None else None

    # ✅ Base General
    BASE_LJ_SEC = int(round(9.25 * 3600))  # Lun-Jue
    BASE_VS_SEC = (6 * 3600) + (53 * 60)  # Sábado (y si quieres usarlo también para viernes NO, ya lo separamos)        # Vie/Sáb

    op = (operator or "").strip()
    is_general = _is_general_operator(op)

    total_hh = 0

    for d in sorted(days):
        day = pd.Timestamp(d)
        day_iso = day.strftime("%Y-%m-%d")
        day_start = pd.Timestamp(f"{day_iso} 00:00:00")
        day_end   = pd.Timestamp(f"{day_iso} 23:59:59.999")

        # verif del día (ya viene filtrado por operaria si aplica)
        if dtv is not None and not dtv.dropna().empty:
            m_v = (dtv >= day_start) & (dtv <= day_end)
            df_v = df_verif.loc[m_v].copy()
            dt_v = dtv.loc[m_v].copy()
        else:
            df_v = df_verif.iloc[0:0].copy()
            dt_v = pd.Series([], dtype="datetime64[ns]")

        # paradas que tocan el día (ya vienen filtradas por operaria según tu pipeline)
        if (
            df_paradas_iv is not None
            and not df_paradas_iv.empty
            and ("dt_start" in df_paradas_iv.columns)
            and ("dt_end" in df_paradas_iv.columns)
        ):
            m_p = (df_paradas_iv["dt_end"] >= day_start) & (df_paradas_iv["dt_start"] <= day_end)
            df_p = df_paradas_iv.loc[m_p].copy()
        else:
            df_p = None

        buckets = _compute_hourly_buckets(
            df_verif=df_v,
            dt_verif=dt_v,
            cols_verif=cols,
            selected_day=day,
            df_paradas_iv=df_p,
        )
        cards = _count_cards(buckets if isinstance(buckets, list) else [])

        if cards <= 0:
            continue

        # -------------------------
        # ¿Cuántas operarias hubo ese día? (sin filtrar)
        # -------------------------
        multi_ops_day = False
        if (not is_general) and (df_verif_all_ops is not None) and (dt_all_ops is not None) and cols.col_nombre:
            m_day_all = dt_all_ops.notna() & (dt_all_ops >= day_start) & (dt_all_ops <= day_end)
            if m_day_all.any() and (cols.col_nombre in df_verif_all_ops.columns):
                ops_day = df_verif_all_ops.loc[m_day_all, cols.col_nombre].astype(str).str.strip()
                uniq_ops = [x for x in ops_day.unique().tolist() if x and x.lower() != "nan"]
                multi_ops_day = (len(uniq_ops) >= 2)   # ✅ TU NUEVA CONDICIÓN (2 o más)

        # =========================================================
        # ✅ SI OPERARIA (no General) Y HUBO 2+ OPERARIAS:
        # Tiempo Pagado = cards*1h - 105(en ventanas fijas)
        # =========================================================
        if (not is_general) and multi_ops_day:
            hh_day = int(cards) * 3600

            # ventanas fijas (105)
            w1_0 = pd.Timestamp(f"{day_iso} 08:30:00")
            w1_1 = pd.Timestamp(f"{day_iso} 08:45:00")  # 15 min
            w2_0 = pd.Timestamp(f"{day_iso} 08:45:00")
            w2_1 = pd.Timestamp(f"{day_iso} 09:00:00")  # 15 min

            w3_0 = pd.Timestamp(f"{day_iso} 13:00:00")
            w3_1 = pd.Timestamp(f"{day_iso} 13:30:00")  # 30 min
            w4_0 = pd.Timestamp(f"{day_iso} 13:30:00")
            w4_1 = pd.Timestamp(f"{day_iso} 14:00:00")  # 30 min

            disc_105 = 0
            disc_105 += _sum_overlap_105_in_window(df_p, w1_0, w1_1)
            disc_105 += _sum_overlap_105_in_window(df_p, w2_0, w2_1)
            disc_105 += _sum_overlap_105_in_window(df_p, w3_0, w3_1)
            disc_105 += _sum_overlap_105_in_window(df_p, w4_0, w4_1)

            hh_day = max(0, hh_day - int(disc_105))
            total_hh += hh_day
            continue

        # -------------------------
        # GENERAL (o día con 1 sola operaria): regla de turno
        # -------------------------
        wd = int(day.dayofweek)  # 0 lun ... 4 vie ... 5 sab ... 6 dom

        if wd in (0, 1, 2, 3):
            extra_h = max(0, int(cards) - 10)
            total_hh += int(BASE_LJ_SEC + extra_h * 3600)
        elif wd == 4:
            # ✅ VIERNES: base 7h. Si hay horas extra (cards>7), sumar extras pero restar 45 min (30+15) una sola vez.
            BASE_FRI_SEC = 7 * 3600
            extra_h = max(0, int(cards) - 7)
            hh_day = int(BASE_FRI_SEC + extra_h * 3600)

            # si hubo horas extra (8,9,...) descontar 45 min una sola vez
            if extra_h > 0:
                hh_day -= (45 * 60)

            total_hh += max(0, hh_day)

        elif wd == 5:
            # ✅ SÁBADO: se mantiene como lo tienes (6:53 base, extras sobre 8)
            extra_h = max(0, int(cards) - 8)
            total_hh += int(BASE_VS_SEC + extra_h * 3600)
        else:
            continue

    # truncar a minuto
    total_hh = (int(total_hh) // 60) * 60
    return int(total_hh)






# ============================================================
#  Entrada para registry
# ============================================================
def get_analysis_entry() -> Dict[str, Any]:
    return {
        "key": ANALYSIS_KEY,
        "title": ANALYSIS_TITLE,
        "fn": analyze_thb,
        "machine": "THB",
        "line": "corte",
    }


# ============================================================
#  Compatibilidad con imports antiguos
# ============================================================
def kpis_thb_corte(
    df_verif,
    *,
    period="day",
    period_value="",
    machine="THB",
    operator="",
    time_start="",
    time_end="",
    df_paradas=None,
):
    return analyze_thb(
        df_verif,
        period=period,
        period_value=period_value,
        machine=machine,
        operator=operator,
        time_start=time_start,
        time_end=time_end,
        df_paradas=df_paradas,
    )

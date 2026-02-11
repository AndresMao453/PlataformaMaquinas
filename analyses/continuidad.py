import pandas as pd
from .registry import register_analysis


def _find_col(df: pd.DataFrame, key: str):
    k = (key or "").strip().lower()
    for c in df.columns:
        if k in str(c).strip().lower():
            return c
    return None


def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True).dt.normalize()


def _coerce_time(s: pd.Series) -> pd.Series:
    # soporta "06:00" o "06:00:00"
    t = pd.to_timedelta(s.astype(str).str.strip(), errors="coerce")
    # si viene datetime/time raro, fallback
    return t


def _mask_by_period(dt: pd.Series, period: str, pv: str) -> pd.Series:
    if dt is None or len(dt) == 0:
        return pd.Series([], dtype=bool)

    period = (period or "day").strip().lower()
    pv = (pv or "").strip()

    m = dt.notna()
    if not pv:
        return m

    if period == "day":
        d = pd.to_datetime(pv, errors="coerce")
        if pd.isna(d):
            return m
        return m & (dt.dt.normalize() == pd.Timestamp(d).normalize())

    if period == "month":
        # pv: YYYY-MM
        try:
            y = int(pv[:4])
            mo = int(pv[5:7])
        except Exception:
            return m
        return m & (dt.dt.year == y) & (dt.dt.month == mo)

    if period == "week":
        # pv: YYYY-W##
        try:
            pv2 = pv.replace("w", "W")
            y = int(pv2.split("-W")[0])
            w = int(pv2.split("-W")[1])
        except Exception:
            return m
        iso = dt.dt.isocalendar()
        return m & (iso["year"].astype(int) == y) & (iso["week"].astype(int) == w)

    return m


def analyze_continuidad(df_hora, df_dia_ref, period="day", period_value="",
                        time_start="", time_end=""):
    import pandas as pd

    def _find_col(df, key):
        k = key.lower()
        for c in df.columns:
            if k in str(c).strip().lower():
                return c
        return None

    def _to_num(s):
        return pd.to_numeric(
            s.astype(str).str.replace(r"[^\d\.\-]", "", regex=True),
            errors="coerce"
        ).fillna(0)

    c_fecha_h = _find_col(df_hora, "fecha")
    c_hora_h = _find_col(df_hora, "hora")
    c_ref_h = _find_col(df_hora, "refer")
    c_total_h = _find_col(df_hora, "total")  # "Total por hora"

    c_fecha_d = _find_col(df_dia_ref, "fecha")
    c_ref_d = _find_col(df_dia_ref, "refer")
    c_qty_d = _find_col(df_dia_ref, "cantidad")  # "Cantidad del día"

    if not all([c_fecha_h, c_hora_h, c_ref_h, c_total_h]):
        return {"status": "bad", "message": "Resumen Hora: faltan columnas (Fecha, Hora, Referencia, Total por hora)."}

    if not all([c_fecha_d, c_ref_d, c_qty_d]):
        return {"status": "bad", "message": "Resumen Día x Ref: faltan columnas (Fecha, Referencia, Cantidad del día)."}

    # ---- normaliza datetime (Fecha + Hora)
    sdt = (df_hora[c_fecha_h].astype(str).str.strip() + " " + df_hora[c_hora_h].astype(str).str.strip())
    dt = pd.to_datetime(sdt, errors="coerce", dayfirst=True)
    df_hora = df_hora.copy()
    df_hora["__dt"] = dt

    # ---- filtro por periodo
    period = (period or "day").lower().strip()
    pv = (period_value or "").strip()

    mask = df_hora["__dt"].notna()
    if pv:
        if period == "day":
            target = pd.to_datetime(pv, errors="coerce")
            if pd.notna(target):
                mask &= (df_hora["__dt"].dt.date == target.date())

        elif period == "month":
            # pv: YYYY-MM
            y = int(pv[:4]); m = int(pv[5:7])
            mask &= (df_hora["__dt"].dt.year == y) & (df_hora["__dt"].dt.month == m)

        elif period == "week":
            iso = df_hora["__dt"].dt.isocalendar()
            # pv: 2026-W05
            import re
            m = re.search(r"(\d{4}).*W(\d{1,2})", pv)
            if m:
                y = int(m.group(1)); w = int(m.group(2))
                mask &= (iso["year"] == y) & (iso["week"] == w)

    dfh = df_hora.loc[mask].copy()
    if dfh.empty:
        return {"status": "bad", "message": "No hay datos para el filtro seleccionado."}

    # ---- numéricos
    dfh["__total_hora"] = _to_num(dfh[c_total_h])

    # ---- KPIs base
    total_producido = int(dfh["__total_hora"].sum())
    horas_activas = int((dfh["__total_hora"] > 0).sum())
    prom_por_hora = (total_producido / horas_activas) if horas_activas else 0

    dias_activos = int(dfh["__dt"].dt.date.nunique())
    refs_unicas = int(dfh[c_ref_h].astype(str).str.strip().nunique())

    # ---- hora pico
    idx = dfh["__total_hora"].idxmax()
    pico_val = int(dfh.loc[idx, "__total_hora"])
    pico_hora = str(dfh.loc[idx, c_hora_h])

    # ---- Serie por hora (para gráfica)
    by_hour = (dfh.groupby(c_hora_h, dropna=False)["__total_hora"]
                 .sum()
                 .sort_index())

    hour_labels = by_hour.index.astype(str).tolist()
    hour_values = [int(x) for x in by_hour.values.tolist()]

    # ---- Top referencias por día (usa df_dia_ref, filtrado SOLO por el mismo periodo)
    dfd = df_dia_ref.copy()
    dfd["__fecha"] = pd.to_datetime(dfd[c_fecha_d], errors="coerce", dayfirst=True)

    if pv:
        if period == "day":
            target = pd.to_datetime(pv, errors="coerce")
            if pd.notna(target):
                dfd = dfd[dfd["__fecha"].dt.date == target.date()]
        elif period == "month":
            y = int(pv[:4]); m = int(pv[5:7])
            dfd = dfd[(dfd["__fecha"].dt.year == y) & (dfd["__fecha"].dt.month == m)]
        elif period == "week":
            iso2 = dfd["__fecha"].dt.isocalendar()
            import re
            mm = re.search(r"(\d{4}).*W(\d{1,2})", pv)
            if mm:
                y = int(mm.group(1)); w = int(mm.group(2))
                dfd = dfd[(iso2["year"] == y) & (iso2["week"] == w)]

    dfd["__qty"] = _to_num(dfd[c_qty_d])
    ref_tot = (dfd.groupby(c_ref_d)["__qty"].sum().sort_values(ascending=False))

    top5 = ref_tot.head(5)
    top_refs = top5.index.astype(str).tolist()
    top_vals = [int(x) for x in top5.values.tolist()]

    top1_share = (top_vals[0] / total_producido) * 100 if (top_vals and total_producido) else 0
    top5_share = (sum(top_vals) / total_producido) * 100 if total_producido else 0

    return {
        "status": "ok",
        "kpis_ui": {
            "Total producido": f"{total_producido:,}".replace(",", "."),
            "Horas activas": str(horas_activas),
            "Prom/hora": f"{prom_por_hora:.1f}",
            "Hora pico": f"{pico_hora} ({pico_val})",
            "Días activos": str(dias_activos),
            "Refs únicas": str(refs_unicas),
            "Top1 %": f"{top1_share:.1f}%",
            "Top5 %": f"{top5_share:.1f}%",
        },
        "series_hour": {
            "labels": hour_labels,
            "values": hour_values
        },
        "top_refs": {
            "labels": top_refs,
            "values": top_vals
        }
    }


register_analysis("continuidad", "Análisis Línea de Continuidad", analyze_continuidad)

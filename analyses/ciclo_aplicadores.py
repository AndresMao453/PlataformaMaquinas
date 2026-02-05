# analyses/ciclo_aplicadores.py
import pandas as pd
from .registry import register_analysis


def analyze_ciclo_aplicadores(
    df: pd.DataFrame,
    period: str = "day",
    period_value: str = "",
    machine: str = "",
) -> dict:
    return {
        "status": "ok",
        "mensaje": "Placeholder Ciclo/Aplicadores. Aquí van KPIs reales.",
        "machine": machine,
        "period": period,
        "period_value": period_value,
        "rows_total": int(len(df)) if df is not None else 0,
        "columns": list(df.columns) if df is not None else [],
    }


register_analysis("ciclo_aplicadores", "Análisis Ciclo/Aplicadores", analyze_ciclo_aplicadores)

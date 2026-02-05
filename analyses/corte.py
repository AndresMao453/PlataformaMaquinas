# analyses/corte.py
from __future__ import annotations
import pandas as pd

from analyses.corte_thb import kpis_thb_corte


def analyze_corte(df_verif: pd.DataFrame, **kwargs):
    machine = (kwargs.get("machine") or "").upper()

    if machine == "THB":
        # df_paradas llega desde app.py
        return kpis_thb_corte(df_verif, **kwargs)

    # HP (más adelante)
    return {
        "status": "pending",
        "message": "HP aún no implementado (estructura distinta).",
        "machine": machine,
    }

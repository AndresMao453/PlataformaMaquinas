# analyses/__init__.py
from __future__ import annotations

from analyses.registry import register_analysis

# Corte
from analyses.corte import analyze_corte

# Aplicación (AP1)
from analyses.aplicacion_excel import analyze_aplicacion1

from . import ciclo_aplicadores  # noqa: F401

register_analysis(
    key="corte",
    title="Análisis Línea de Corte",
    fn=analyze_corte,
)

register_analysis(
    key="aplicacion",
    title="Análisis Línea de Aplicación (AP1)",
    fn=analyze_aplicacion1,
)

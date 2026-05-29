# tools/ajustar_modelo_hp.py
from __future__ import annotations

import numpy as np
import pandas as pd

# Modelo:
#   T_lote = c + piezas * (a + b * longitud_mm)

DATOS_HP = [
    # longitud_mm, tiempo_lote_s, piezas, sin_terminal, terminal_1, terminal_2
    [793, 38.88, 5, "si", "no", "no"],
    [1113, 40.68, 5, "si", "no", "no"],
    [1268, 47.16, 5, "no", "si", "no"],
    [3879, 81.90, 5, "no", "si", "no"],
    [4379, 87.90, 5, "no", "si", "no"],
    [9184, 140.28, 5, "no", "si", "no"],
    [9344, 145.56, 5, "no", "si", "no"],
    [9385, 148.86, 5, "no", "si", "no"],
    [4069, 78.84, 5, "no", "si", "no"],
    [7634, 138.36, 5, "no", "si", "si"],
    [8634, 155.88, 5, "no", "si", "si"],
    [7594, 135.72, 5, "no", "si", "si"],
    [1863, 59.64, 5, "no", "si", "si"],
    [1863, 62.10, 5, "no", "si", "si"],
    [6732, 125.04, 5, "no", "si", "si"],
    [6692, 123.12, 5, "no", "si", "si"],
    [111, 66.30, 10, "no", "si", "no"],
    [380, 66.30, 10, "no", "si", "no"],
    [593, 60.00, 10, "si", "no", "no"],
    [718, 66.84, 10, "si", "no", "no"],
    [1418, 75.84, 10, "si", "no", "no"],
    [2589, 107.88, 10, "si", "no", "no"],
    [3264, 114.18, 9, "si", "no", "no"],
    [5367, 172.38, 10, "si", "no", "no"],
    [6802, 227.88, 10, "no", "si", "si"],
    [6842, 229.38, 10, "no", "si", "si"],
]

def _si(x: str) -> bool:
    return str(x or "").strip().lower() in ("si", "sí", "s", "1", "true")


def clasificar_tipo(row: pd.Series) -> int:
    if _si(row["sin_terminal"]):
        return 1
    if _si(row["terminal_1"]) and _si(row["terminal_2"]):
        return 3
    return 2


def ajustar_modelo_por_tipo(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    resultados = {}

    for tipo, grupo in df.groupby("tipo"):
        longitud = grupo["longitud_mm"].to_numpy(dtype=float)
        piezas = grupo["piezas"].to_numpy(dtype=float)
        tiempo = grupo["tiempo_lote_s"].to_numpy(dtype=float)

        # T_lote = c + piezas*a + piezas*longitud*b
        x = np.column_stack([np.ones(len(grupo)), piezas, piezas * longitud])
        coef, *_ = np.linalg.lstsq(x, tiempo, rcond=None)
        c, a, b = coef

        estimado = x @ coef
        error = tiempo - estimado
        rmse = float(np.sqrt(np.mean(error ** 2)))
        ss_res = float(np.sum(error ** 2))
        ss_tot = float(np.sum((tiempo - np.mean(tiempo)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        resultados[int(tipo)] = {
            "c": float(c),
            "a": float(a),
            "b": float(b),
            "rmse": rmse,
            "r2": r2,
            "registros": int(len(grupo)),
        }

        df.loc[grupo.index, "tiempo_estimado_s"] = estimado
        df.loc[grupo.index, "error_s"] = error
        df.loc[grupo.index, "tc_real_s_pieza"] = tiempo / piezas
        df.loc[grupo.index, "tc_estimado_s_pieza"] = estimado / piezas

    return resultados


def imprimir_hp_fit_lote(resultados: dict[int, dict[str, float]]) -> None:
    print("\nHP_FIT_LOTE = {")
    nombres = {1: "sin terminal", 2: "1 terminal", 3: "2 terminales"}
    for tipo in (1, 2, 3):
        r = resultados.get(tipo, {"c": 0.0, "a": 0.0, "b": 0.0})
        print(f"    # Tipo {tipo}: {nombres[tipo]}")
        print(f'    {tipo}: {{"c": {r["c"]:.6f}, "a": {r["a"]:.6f}, "b": {r["b"]:.9f}}},')
        if tipo != 3:
            print()
    print("}")


def main() -> None:
    df = pd.DataFrame(
        DATOS_HP,
        columns=["longitud_mm", "tiempo_lote_s", "piezas", "sin_terminal", "terminal_1", "terminal_2"],
    )
    df["tipo"] = df.apply(clasificar_tipo, axis=1)
    resultados = ajustar_modelo_por_tipo(df)
    imprimir_hp_fit_lote(resultados)

    print("\nResumen del ajuste")
    print("=" * 70)
    for tipo in sorted(resultados):
        r = resultados[tipo]
        print(
            f"Tipo {tipo}: registros={r['registros']} | "
            f"RMSE={r['rmse']:.4f} s | R2={r['r2']:.4f} | "
            f"c={r['c']:.6f}, a={r['a']:.6f}, b={r['b']:.9f}"
        )

    print("\nValidación")
    print("=" * 70)
    print(
        df[[
            "longitud_mm", "piezas", "tipo", "tiempo_lote_s",
            "tiempo_estimado_s", "error_s", "tc_real_s_pieza", "tc_estimado_s_pieza"
        ]].sort_values(["tipo", "longitud_mm", "piezas"]).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()

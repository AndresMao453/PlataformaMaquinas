# config.py
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"xlsx", "xls"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

# Cambia esto si quieres (pero NO lo publiques en GitHub si es real)
SECRET_KEY = "dev-secret-key-change-me"

# =========================
# Google Sheets - Aplicación / Unión
# =========================
GSHEET_ID = "1xCx3fegK_Xm4dXQnCvFIBDh0AHsKr6yIjdsSqG-YdpE"
GSHEET_GID_APLICACION = 1283368656
GSHEET_GID_UNION = 1848654694
GSHEET_TTL_S = 600
APP_ANALYSIS_TTL_S = 600
APP_CACHE_MAX_ITEMS = 120


# =========================
# Google Sheets - Aplicación / Unión (Máquina 2)
# =========================
GSHEET_ID2 = "1-He6-tg5l5N1ayf4bDQz1aHAgMcmm12bOIjbXGrJ05I"
GSHEET_GID_APLICACION2 = 1876028636
GSHEET_GID_UNION2 = 536757828

# =========================
# APLICACIÓN/UNIÓN — Unión Máquina 1
# =========================
GSHEET_ID3 = "1Xb7-K0mmaDNDb4k1jx3TYKJKmJbpMy_EqGcxuffP_cw"
GSHEET_GID_APLICACION3 = 93019910
GSHEET_GID_UNION3 = 1190305223

# Opcional: si existe una pestaña RESUMEN_TIEMPOS en Unión M1, coloca aquí su GID.
# Si se deja en 0, Unión M1 usa el TC histórico y evita descargar el XLSX completo.
GSHEET_GID_RESUMEN_TIEMPOS3 = 0

# UNION - Máquina 2
GSHEET_ID4 = "1-He6-tg5l5N1ayf4bDQz1aHAgMcmm12bOIjbXGrJ05I"
GSHEET_GID_APLICACION4 = 1876028636
GSHEET_GID_UNION4 = 536757828

# =========================
# APLICACIÓN — Máquina 2
# =========================
GSHEET_ID5 = "1i_Vxjnzpg216S77AtSOclyfe9tIoM_GITqwonOjErgM"
GSHEET_GID_APLICACION5 = 800349176
GSHEET_GID_UNION5 = 847861322

# ===== Aliases compat (aplicacion_excel.py viejo) =====
APLICACION_SHEET_ID = GSHEET_ID
APLICACION_GID_APLICACION = GSHEET_GID_APLICACION
APLICACION_GID_UNION = GSHEET_GID_UNION
APLICACION_SHEET_TTL_S = GSHEET_TTL_S


# =========================
# Google Sheets - Corte HP
# =========================
HP_SHEET_ID = "1RcogFd-F4tq7y5d587hsHIE6GsJQipP5HrdxFnNbhcs"

HP_GID_VERIF = 523940500   # "Verificacion de Medidas"
HP_GID_PARADAS = 38816505  # "Paradas"
HP_GID_CORTE = 464207492    # "Corte"


# =========================
# Google Sheets - Corte THB
# =========================
THB_SHEET_ID = "1LWP9Z4AfBQFPOX3mydgTNPUmAEhsiWc9PSVp78-bLvA"

THB_GID_VERIF = 145813485      # "Verificación de Medidas"
THB_GID_PARADAS = 1720625606   # "Paradas"


# =========================
# CONTINUIDAD (GOOGLE SHEETS)
# =========================
CONT_BUSES_SHEET_ID = "13pRTp1-HWVvdfNGymzhbS2v4iDoojMGuPVcZUdwDEPk"
CONT_BUSES_GID_RESUMEN_HORA = 855023329
CONT_BUSES_GID_RESUMEN_DIA_REF = 1764533798


# TTL (si ya tienes GSHEET_TTL_S, no agregues esto)
# GSHEET_TTL_S = 10

# ✅ Tokens (deben existir como constantes globales)
GSHEET_CONT_BUSES_TOKEN = "__GSHEET_CONT_BUSES__"
GSHEET_CONT_MOTOS_TOKEN = "__GSHEET_CONT_MOTOS__"

# =========================
# Unión 1 - Ponderados dinámicos por combinación
# =========================
# Estos parámetros aplican SOLO para UNION - Máquina 1.
# La combinación dinámica es: Moto + Masa + Cantidad de cables.
# Unión 2 queda independiente y no reutiliza estos ponderados.
CRIMPADO_MIX_MIN_REGISTROS = 5
CRIMPADO_MIX_MIN_PULSOS = 50
CRIMPADO_MIX_TC_MIN_S = 3.0
CRIMPADO_MIX_TC_MAX_S = 180.0



# =========================
# Unión 2 — Modo del TC por hora
# =========================
# Define contra qué estándar se calcula el tiempo de ciclo por hora de Unión 2:
#   "meta"     -> usa el MEJOR ciclo observado de cada referencia (recomendado).
#   "promedio" -> usa el promedio ponderado histórico de cada referencia.
# La lógica del TC por hora (duración ÷ planeado, por referencia, sin mezclar)
# NO cambia con esto; solo cambia el estándar de comparación de cada referencia.
UNION2_TC_MODO = "meta"


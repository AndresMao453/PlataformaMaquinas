# config.py
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"xlsx", "xls"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

# Cambia esto si quieres (pero NO lo publiques en GitHub si es real)
SECRET_KEY = "dev-secret-key-change-me"

# =========================
# Google Sheets - Aplicación / Unión
# =========================
GSHEET_ID = "16WX3jqnbmYCglR2jH_1uHyWH-DKFuh80uEynKVghnyE"
GSHEET_GID_APLICACION = 1418050425
GSHEET_GID_UNION = 369782990
GSHEET_TTL_S = 120


# =========================
# Google Sheets - Aplicación / Unión (Máquina 2)
# =========================
GSHEET_ID2 = "1XE2V1NLxvlvGJodcjp70F4GzDokx-usn1nI0pmFQhjI"
GSHEET_GID_APLICACION2 = 1822604115
GSHEET_GID_UNION2 = 41529279

# =========================
# APLICACIÓN/UNIÓN — Unión Máquina 1
# =========================
GSHEET_ID3 = "1xC4dIaUKo0IYyS232HuIPAzI0DeksnhOvsAX57rGzko"
GSHEET_GID_APLICACION3 = 326889329
GSHEET_GID_UNION3 = 1786033126

# UNION - Máquina 2
GSHEET_ID4 = "1D_EWkHpk8VDnBle35Ptr_x6AU07TGNv_ocUimLo13K0"
GSHEET_GID_APLICACION4 = 883345468
GSHEET_GID_UNION4 = 87676947

# ===== Aliases compat (aplicacion_excel.py viejo) =====
APLICACION_SHEET_ID = GSHEET_ID
APLICACION_GID_APLICACION = GSHEET_GID_APLICACION
APLICACION_GID_UNION = GSHEET_GID_UNION
APLICACION_SHEET_TTL_S = GSHEET_TTL_S


# =========================
# Google Sheets - Corte HP
# =========================
HP_SHEET_ID = "1RcogFd-F4tq7y5d587hsHIE6GsJQipP5HrdxFnNbhcs"

HP_GID_VERIF = 941764487   # "Verificacion de Medidas"
HP_GID_PARADAS = 866145386  # "Paradas"
HP_GID_CORTE = 738815404    # "Corte"


# =========================
# Google Sheets - Corte THB
# =========================
THB_SHEET_ID = "1LWP9Z4AfBQFPOX3mydgTNPUmAEhsiWc9PSVp78-bLvA"

THB_GID_VERIF = 145813485      # "Verificación de Medidas"
THB_GID_PARADAS = 1720625606   # "Paradas"

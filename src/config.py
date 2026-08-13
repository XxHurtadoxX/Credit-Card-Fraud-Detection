"""Rutas, semillas y parametros economicos del proyecto.

Todo lo que un lector podria querer cambiar vive aqui, no disperso en el codigo.
Los supuestos de costo estan documentados en README.md y son deliberadamente
discutibles: cambiarlos cambia la decision, y ese es justamente el punto.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = DATA_DIR / "creditcard.csv"
MODEL_PATH = MODELS_DIR / "inference_pipeline.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"

RANDOM_STATE = 42
TARGET = "Class"

# Fraccion final de la ventana temporal reservada para prueba.
TEST_FRACTION = 0.20
# Dentro del tramo de entrenamiento, fraccion final usada para fijar el umbral.
VALIDATION_FRACTION = 0.25
CV_SPLITS = 4

# Parametros del modelo de costos, en la misma unidad monetaria que Amount.
#
# COST_REVIEW      lo que cuesta que un analista mire una alerta.
# COST_FRICTION    costo de bloquear una compra legitima: llamada, reposicion
#                  de tarjeta, abandono del cliente. Es el numero mas incierto
#                  de los tres y el que mas mueve el resultado.
# RECOVERY_RATE    fraccion del monto que se recupera cuando el fraude se
#                  detecta a tiempo. Nunca es 1: parte del dinero ya se fue.
# CHARGEBACK_FEE   cargo fijo del emisor por cada contracargo procesado.
COST_REVIEW = 4.0
COST_FRICTION = 25.0
RECOVERY_RATE = 0.85
CHARGEBACK_FEE = 15.0

# Tope de monto al construir los pesos de la perdida sensible al costo. Sin el,
# una sola operacion grande domina el gradiente de un fold entero. No afecta el
# calculo del costo reportado, solo el peso durante el entrenamiento.
AMOUNT_CAP = 1000.0

# Alertas que el equipo de revision puede procesar por dia. Restringe que
# umbrales son operativamente viables, no solo que umbrales son optimos.
#
# El dataset promedia unas 142000 transacciones y 246 fraudes por dia, asi que
# 400 alertas equivalen a revisar el 0.28 por ciento del flujo. Sirve de tope
# para descartar umbrales que inundan la cola de revision, no como restriccion
# principal: el optimo economico suele quedar bastante por debajo.
DAILY_REVIEW_CAPACITY = 400

# Paleta de marca. Cobre para valor, patina para senal.
COLOR_INK = "#0A0E14"
COLOR_COPPER = "#8A4B1E"
COLOR_PATINA = "#0E7C68"
COLOR_LINE = "#E3E7EC"
COLOR_MID = "#5B6673"
COLOR_DANGER = "#DC2626"

"""Carga del dataset y particion temporal.

El corte se hace por tiempo. El modelo se entrena con el pasado y se evalua con
el futuro, que es la forma en que va a operar una vez desplegado.

Los fraudes llegan en rachas, con varias operaciones seguidas sobre la misma
tarjeta comprometida. Un corte al azar puede dejar una racha repartida entre los
dos conjuntos, y entonces la prueba mide sobre casos de los que el modelo ya vio
un gemelo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


def load_raw(path=None) -> pd.DataFrame:
    """Devuelve el dataset crudo ordenado por tiempo.

    Si el CSV no esta en disco lo descarga desde Kaggle. La descarga necesita
    credenciales configuradas; ver README.
    """
    path = path or config.RAW_CSV

    if not path.exists():
        log.info("No hay CSV local, descargando %s", config.KAGGLE_DATASET)
        import kagglehub

        # kagglehub devuelve un directorio; el archivo siempre se llama igual.
        source = Path(kagglehub.dataset_download(config.KAGGLE_DATASET)) / "creditcard.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(source).to_csv(path, index=False)

    df = pd.read_csv(path)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    log.info("Cargadas %s transacciones, %s fraudes", f"{len(df):,}",
             int(df[config.TARGET].sum()))
    return df


def window_days(times: pd.Series) -> float:
    """Duracion en dias del tramo cubierto por una serie de `Time`."""
    if times.empty:
        return 0.0
    return float(times.max() - times.min()) / SECONDS_PER_DAY


def temporal_split(df: pd.DataFrame, test_fraction: float | None = None):
    """Corta el dataframe en pasado y futuro por la columna `Time`."""
    test_fraction = config.TEST_FRACTION if test_fraction is None else test_fraction
    cut = int(len(df) * (1 - test_fraction))
    boundary = df["Time"].iloc[cut]

    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)

    log.info(
        "Corte temporal en t=%.0f s. Entrenamiento %s filas / %s fraudes, "
        "prueba %s filas / %s fraudes",
        boundary,
        f"{len(train):,}", int(train[config.TARGET].sum()),
        f"{len(test):,}", int(test[config.TARGET].sum()),
    )
    return train, test


def split_xy(df: pd.DataFrame):
    """Separa predictores, etiqueta y monto.

    El monto se devuelve aparte porque el modelo no lo usa en bruto pero el
    modelo de costos si lo necesita.
    """
    y = df[config.TARGET].astype(int)
    amount = df["Amount"].astype(float)
    X = df.drop(columns=[config.TARGET])
    return X, y, amount

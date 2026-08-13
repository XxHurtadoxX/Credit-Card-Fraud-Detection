"""Puntuacion de transacciones con el modelo entrenado.

    python -m src.predict --input transacciones.csv --output alertas.csv

El artefacto guardado incluye el umbral que se eligio en validacion. Cargarlo
junto al pipeline evita el problema clasico de tener el umbral escrito a mano en
el codigo de servicio y otro distinto en el notebook.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd

from . import config, features

_bundle = None


def load_bundle(path=None) -> dict:
    """Carga el pipeline y su umbral. Cachea para no releer en cada llamada."""
    global _bundle
    if _bundle is not None:
        return _bundle

    # En un endpoint administrado de Azure ML el modelo se monta en una ruta que
    # el servicio comunica por variable de entorno.
    if path is None:
        model_dir = os.environ.get("AZUREML_MODEL_DIR")
        path = Path(model_dir) / config.MODEL_PATH.name if model_dir else config.MODEL_PATH

    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(
            f"{path} no tiene el formato esperado. Ejecuta python -m src.train "
            "para regenerarlo."
        )
    _bundle = bundle
    return _bundle


def score(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Devuelve probabilidad y decision para cada fila."""
    bundle = bundle or load_bundle()
    X = features.build_features(df)
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    threshold = bundle["threshold"]

    return pd.DataFrame({
        "fraud_probability": proba.round(6),
        "alert": (proba >= threshold).astype(int),
        "threshold": threshold,
    }, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="Puntua un CSV de transacciones")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = score(df, load_bundle(args.model))

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"{len(result):,} filas puntuadas -> {args.output}")
    else:
        print(result.head(20).to_string())

    print(f"Alertas: {int(result['alert'].sum()):,} de {len(result):,} "
          f"(umbral {result['threshold'].iloc[0]:g})")


if __name__ == "__main__":
    main()

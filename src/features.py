"""Construccion de predictores.

Treinta y una columnas. Las 28 componentes principales, el logaritmo del monto y
la hora del dia descompuesta en seno y coseno.

`Time` no entra como predictor. Mide segundos desde la primera transaccion del
archivo, de modo que su valor solo existe dentro de esta ventana de dos dias. Un
arbol que aprende que hay mas fraude despues del segundo 130000 memoriza el
archivo y no el fenomeno. La hora del dia si se repite manana, y entra en seno y
coseno para que las 23:00 y la 01:00 queden cerca en lugar de en extremos
opuestos.

Tampoco se generan derivadas de las componentes principales. Un arbol reproduce
cualquier transformacion monotona con un corte y una region no monotona con dos,
asi que el valor absoluto o el cuadrado de una componente no agregan informacion
y reparten la importancia entre columnas casi identicas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
ENGINEERED = ["Amount_log", "hour_sin", "hour_cos"]
FEATURE_COLUMNS = PCA_COLUMNS + ENGINEERED


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma el dataframe crudo en la matriz de predictores.

    La funcion opera fila por fila y no estima nada a partir del conjunto, asi
    que aplicarla antes de partir los datos no filtra informacion. El escalado,
    que si necesita estimar medianas y rangos, vive dentro del pipeline y por
    lo tanto se ajusta solo con los datos de entrenamiento de cada fold.
    """
    out = pd.DataFrame(index=df.index)

    for col in PCA_COLUMNS:
        out[col] = df[col].astype(float)

    # El monto es muy asimetrico: mediana cerca de 22 y cola hasta 25000.
    out["Amount_log"] = np.log1p(df["Amount"].astype(float))

    hour = (df["Time"].astype(float) % SECONDS_PER_DAY) / 3600.0
    angle = 2 * np.pi * hour / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)

    return out[FEATURE_COLUMNS]

"""Lineas base previas al aprendizaje automatico.

Un emisor sin modelo estadistico no aprueba todo. Opera un motor de reglas
escritas por analistas, y ese motor es la vara que un modelo tiene que superar
para justificar su costo. Comparar contra "no hacer nada" mide el valor de tener
cualquier control, no el valor del modelo.

Se implementan tres reglas de complejidad creciente, todas legibles por una
persona y todas calibradas sobre el mismo tramo de validacion que usa el modelo.

    monto           revisar toda operacion que pase de un monto
    monto_hora      el monto, con un corte mas bajo en la madrugada
    arbol           motor de reglas de profundidad 3, hasta ocho hojas

El arbol de decision es el equivalente honesto de un motor de reglas maduro. Se
ajusta con datos, igual que las reglas de un area de riesgo se calibran con el
historico de contracargos, y el resultado se puede leer como una lista de
condiciones.

Las componentes PCA quedan disponibles para el arbol pero no para las dos reglas
manuales, porque un analista no escribe umbrales sobre variables anonimizadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config

NIGHT_HOURS = (0, 6)


def _hour(times) -> np.ndarray:
    return (np.asarray(times, dtype=float) % 86_400) / 3600.0


def amount_rule_scores(amount, cut: float) -> np.ndarray:
    """Uno si el monto pasa el corte. Es la regla mas antigua del oficio."""
    return (np.asarray(amount, dtype=float) >= cut).astype(float)


def amount_hour_rule_scores(amount, times, cut: float, night_cut: float) -> np.ndarray:
    """El corte de monto, mas bajo en la madrugada.

    La madrugada concentra una tasa de fraude varias veces superior al promedio,
    de modo que bajar el umbral en esa franja es la primera correccion que un
    analista agrega sobre la regla de monto.
    """
    amount = np.asarray(amount, dtype=float)
    hour = _hour(times)
    night = (hour >= NIGHT_HOURS[0]) & (hour < NIGHT_HOURS[1])
    return np.where(night, amount >= night_cut, amount >= cut).astype(float)


def fit_rule_engine(X, y, max_depth: int = 3) -> DecisionTreeClassifier:
    """Motor de reglas, un arbol lo bastante corto para leerse en voz alta."""
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        min_samples_leaf=50,
        random_state=config.RANDOM_STATE,
    )
    tree.fit(X, y)
    return tree


def rule_engine_text(tree, feature_names) -> str:
    return export_text(tree, feature_names=list(feature_names), decimals=2)


def _best_over_grid(grid, score_fn, y, amount, window_days):
    """Elige el parametro de la regla que minimiza el costo."""
    from . import economics

    best = None
    for params in grid:
        scores = score_fn(*params)
        outcome = economics.cost_at_threshold(y, scores, amount, 0.5, window_days)
        if best is None or outcome.savings > best[1].savings:
            best = (params, outcome)
    return best


def calibrate(X, y, amount, times, window_days, tree=None) -> dict:
    """Calibra las tres reglas sobre un tramo y devuelve su resultado.

    Se llama sobre validacion, igual que el umbral del modelo, para que la
    comparacion sea entre iguales.
    """
    from . import economics

    amount = np.asarray(amount, dtype=float)
    quantiles = np.quantile(amount, np.arange(0.90, 0.9999, 0.002))
    cuts = np.unique(np.round(quantiles, 2))

    out = {}

    params, outcome = _best_over_grid(
        [(c,) for c in cuts],
        lambda c: amount_rule_scores(amount, c),
        y, amount, window_days,
    )
    out["monto"] = {
        "descripcion": f"revisar si Amount >= {params[0]:.2f}",
        "parametros": {"cut": round(float(params[0]), 2)},
        **outcome.as_dict(),
    }

    night_cuts = np.unique(np.round(np.quantile(amount, np.arange(0.70, 0.999, 0.01)), 2))
    params, outcome = _best_over_grid(
        [(c, n) for c in cuts[::2] for n in night_cuts[::2]],
        lambda c, n: amount_hour_rule_scores(amount, times, c, n),
        y, amount, window_days,
    )
    out["monto_hora"] = {
        "descripcion": (f"revisar si Amount >= {params[0]:.2f}, "
                        f"o si Amount >= {params[1]:.2f} entre las 0 y las 6"),
        "parametros": {"cut": round(float(params[0]), 2),
                       "night_cut": round(float(params[1]), 2)},
        **outcome.as_dict(),
    }

    if tree is not None:
        proba = tree.predict_proba(X)[:, 1]
        chosen = economics.pick_threshold(y, proba, amount, window_days)
        out["arbol"] = {
            "descripcion": f"motor de reglas de profundidad {tree.get_depth()}, "
                           f"{tree.get_n_leaves()} hojas, umbral {chosen.threshold}",
            "parametros": {"max_depth": int(tree.get_depth()),
                           "leaves": int(tree.get_n_leaves()),
                           "threshold": chosen.threshold},
            **chosen.as_dict(),
        }

    return out


def apply(name: str, params: dict, X, amount, times, tree=None) -> np.ndarray:
    """Reaplica una regla ya calibrada a otro tramo."""
    if name == "monto":
        return amount_rule_scores(amount, params["cut"])
    if name == "monto_hora":
        return amount_hour_rule_scores(amount, times, params["cut"], params["night_cut"])
    if name == "arbol":
        return tree.predict_proba(X)[:, 1]
    raise ValueError(f"Regla desconocida: {name}")


def best_rule(calibration: dict) -> tuple[str, dict]:
    """La regla de mayor ahorro. Es la vara que el modelo debe superar."""
    name = max(calibration, key=lambda k: calibration[k]["savings"])
    return name, calibration[name]


def uplift(model_savings: float, rule_savings: float) -> dict:
    """Cuanto agrega el modelo sobre el motor de reglas."""
    return {
        "model_savings": round(float(model_savings), 2),
        "rule_savings": round(float(rule_savings), 2),
        "absolute": round(float(model_savings - rule_savings), 2),
        "relative": round(float(model_savings / rule_savings - 1), 4)
        if rule_savings > 0 else None,
    }

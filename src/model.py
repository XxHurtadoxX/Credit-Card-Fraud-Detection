"""Candidatos y seleccion por ahorro esperado.

Cada candidato se distingue por su funcion de perdida, porque es ahi donde se
declara que le importa al banco. Se comparan cinco.

    logreg_balanced     referencia lineal, log loss con pesos de clase
    xgb_plain           log loss sin ponderar
    xgb_weighted        log loss con pesos de clase, ignora el monto
    xgb_cost            log loss ponderada por arrepentimiento, lineal en monto
    xgb_cost_balanced   igual, reequilibrando el peso total de cada clase

La referencia lineal cumple una funcion. Las componentes principales del dataset
ya separan bastante las dos clases, asi que sin una vara lineal no se puede
saber cuanta de la separacion aporta el modelo y cuanta venia en los datos.

El criterio de seleccion es el ahorro que produce cada perdida en el punto de
operacion, sujeto a la capacidad de revision. El area bajo la curva de precision
y recall se reporta al lado porque hace comparable el proyecto con la literatura
del dataset, pero no decide.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from . import config
from .costs import CostWeightedPipeline

log = logging.getLogger(__name__)

def _suggest(trial) -> dict:
    """Propone una combinacion de hiperparametros para XGBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }


def param_prefix(pipeline) -> str:
    """Donde viven los hiperparametros de XGBoost dentro del pipeline.

    Los candidatos sensibles al costo envuelven al pipeline completo, de modo que
    el clasificador queda un nivel mas abajo.
    """
    return ("estimator__clf__" if isinstance(pipeline, CostWeightedPipeline)
            else "clf__")


def _xgb(scale_pos_weight: float | None = None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def imbalance_ratio(y) -> float:
    """Negativos por cada positivo. Es el valor natural de scale_pos_weight."""
    y = np.asarray(y)
    positives = int((y == 1).sum())
    return float((y == 0).sum() / positives) if positives else 1.0


def candidates(y_train) -> dict[str, Pipeline]:
    """Los pipelines que compiten. El escalado siempre va dentro del pipeline."""
    ratio = imbalance_ratio(y_train)

    return {
        "logreg_balanced": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=config.RANDOM_STATE,
            )),
        ]),
        "xgb_plain": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", _xgb()),
        ]),
        "xgb_weighted": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", _xgb(scale_pos_weight=ratio)),
        ]),
        "xgb_cost": CostWeightedPipeline(
            estimator=Pipeline([("scaler", RobustScaler()), ("clf", _xgb())]),
            scheme="linear", amount_cap=config.AMOUNT_CAP,
        ),
        "xgb_cost_balanced": CostWeightedPipeline(
            estimator=Pipeline([("scaler", RobustScaler()), ("clf", _xgb())]),
            scheme="balanced", amount_cap=config.AMOUNT_CAP,
        ),
    }


def temporal_cv(n_splits: int | None = None) -> TimeSeriesSplit:
    """Validacion cruzada que respeta el orden temporal.

    `TimeSeriesSplit` entrena siempre con datos anteriores al fold de
    validacion. Un `StratifiedKFold` sobre datos ordenados en el tiempo dejaria
    que el modelo aprenda del futuro en tres de cada cuatro folds.
    """
    return TimeSeriesSplit(n_splits=n_splits or config.CV_SPLITS)


def cross_val_ap(pipeline, X, y, cv=None) -> tuple[float, float, list[float]]:
    """Average precision por fold, calculada a mano para poder inspeccionarla.

    `cross_val_score` serviria igual, pero cuando un fold da un valor absurdo
    conviene poder ver los folds uno por uno en lugar de un promedio.
    """
    cv = cv or temporal_cv()
    scores = []
    for train_idx, val_idx in cv.split(X):
        fold = _clone_and_fit(pipeline, X, y, train_idx)
        if fold is None:
            scores.append(float("nan"))
            continue
        proba = fold.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(y.iloc[val_idx], proba))

    clean = [s for s in scores if not np.isnan(s)]
    mean = float(np.mean(clean)) if clean else float("nan")
    std = float(np.std(clean)) if clean else float("nan")
    return mean, std, scores


def _clone_and_fit(pipeline, X, y, train_idx):
    y_fold = y.iloc[train_idx]
    if y_fold.nunique() < 2:
        log.warning("Fold sin ambas clases, se omite")
        return None
    model = clone(pipeline)
    model.fit(X.iloc[train_idx], y_fold)
    return model


def cross_val_savings(pipeline, X, y, amount, times, cv=None) -> tuple[float, float, list]:
    """Ahorro por fold, con el umbral elegido dentro de cada fold.

    Elegir el umbral en el mismo fold donde se mide sobreestima, asi que cada
    fold se parte en dos. El modelo se ajusta con la primera parte del tramo de
    entrenamiento, el umbral se fija con la segunda, y el ahorro se mide en el
    tramo de validacion del fold.
    """
    from . import economics

    cv = cv or temporal_cv()
    shares, value_recalls, alert_rates = [], [], []
    for train_idx, val_idx in cv.split(X):
        inner = int(len(train_idx) * (1 - config.VALIDATION_FRACTION))
        fit_idx, thr_idx = train_idx[:inner], train_idx[inner:]
        if len(thr_idx) == 0 or y.iloc[fit_idx].nunique() < 2:
            continue

        fold = _clone_and_fit(pipeline, X, y, fit_idx)
        if fold is None:
            continue

        thr_days = _span_days(times.iloc[thr_idx])
        chosen = economics.pick_threshold(
            y.iloc[thr_idx],
            fold.predict_proba(X.iloc[thr_idx])[:, 1],
            amount.iloc[thr_idx].to_numpy(),
            thr_days,
        )

        val_days = _span_days(times.iloc[val_idx])
        outcome = economics.cost_at_threshold(
            y.iloc[val_idx],
            fold.predict_proba(X.iloc[val_idx])[:, 1],
            amount.iloc[val_idx].to_numpy(),
            chosen.threshold,
            val_days,
        )
        shares.append(outcome.savings_share)
        value_recalls.append(outcome.value_recall)
        alert_rates.append(outcome.alerts_per_day)

    return {
        "savings_share_mean": float(np.mean(shares)) if shares else float("nan"),
        "savings_share_std": float(np.std(shares)) if shares else float("nan"),
        "savings_share_folds": shares,
        "value_recall_mean": float(np.mean(value_recalls)) if value_recalls else float("nan"),
        "alerts_per_day_mean": float(np.mean(alert_rates)) if alert_rates else float("nan"),
    }


def _span_days(times) -> float:
    if len(times) == 0:
        return 0.0
    return max(float(times.max() - times.min()) / 86_400, 1e-6)


def select_candidate(X, y, amount, times, cv=None) -> tuple[str, dict]:
    """Compara las perdidas por ahorro esperado y devuelve la mejor.

    El average precision se calcula y se guarda al lado, pero el criterio que
    decide es el ahorro, porque es la cantidad que le importa al banco.
    """
    cv = cv or temporal_cv()
    results = {}
    for name, pipeline in candidates(y).items():
        ap_mean, ap_std, _ = cross_val_ap(pipeline, X, y, cv)
        sv = cross_val_savings(pipeline, X, y, amount, times, cv)
        results[name] = {
            "savings_share_mean": round(sv["savings_share_mean"], 4),
            "savings_share_std": round(sv["savings_share_std"], 4),
            "savings_share_folds": [round(f, 4) for f in sv["savings_share_folds"]],
            "value_recall_mean": round(sv["value_recall_mean"], 4),
            "alerts_per_day_mean": round(sv["alerts_per_day_mean"], 1),
            # Secundaria, para comparar con la literatura del dataset.
            "ap_mean": round(ap_mean, 4),
            "ap_std": round(ap_std, 4),
        }
        log.info("%-18s ahorro %.1f%% +/- %.1f | valor %.1f%% | %.0f alertas/dia",
                 name, sv["savings_share_mean"] * 100, sv["savings_share_std"] * 100,
                 sv["value_recall_mean"] * 100, sv["alerts_per_day_mean"])

    best = max(results, key=lambda k: results[k]["savings_share_mean"])
    log.info("Perdida seleccionada: %s", best)
    return best, results


def tune(pipeline, X, y, amount, times, n_trials: int = 40, cv=None) -> dict:
    """Optimiza los hiperparametros contra el ahorro esperado.

    El muestreador propone cada combinacion en funcion de las anteriores, de modo
    que concentra las pruebas en la region del espacio que viene rindiendo. El
    objetivo es el mismo que decide la seleccion de la perdida, asi que el ajuste
    fino optimiza la cantidad que le importa al banco y no una metrica sustituta.

    Devuelve el pipeline ya ajustado con la mejor combinacion, junto al detalle
    del estudio.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv = cv or temporal_cv()
    prefix = param_prefix(pipeline)

    def objective(trial):
        candidate = clone(pipeline)
        candidate.set_params(**{f"{prefix}{k}": v for k, v in _suggest(trial).items()})
        return cross_val_savings(candidate, X, y, amount, times, cv)["savings_share_mean"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = clone(pipeline)
    best.set_params(**{f"{prefix}{k}": v for k, v in study.best_params.items()})
    best.fit(X, y)

    try:
        importances = {k: round(float(v), 4) for k, v in
                       optuna.importance.get_param_importances(study).items()}
    except Exception:
        # La importancia necesita variacion entre pruebas; si todas rinden igual
        # no se puede calcular y el resto del reporte no depende de ella.
        importances = {}

    log.info("Mejor ahorro en la busqueda: %.1f%% en %s pruebas",
             study.best_value * 100, n_trials)
    log.info("Mejores hiperparametros: %s", study.best_params)

    return {
        "estimator": best,
        "best_value": round(float(study.best_value), 4),
        "best_params": {k: _plain(v) for k, v in study.best_params.items()},
        "n_trials": n_trials,
        "param_importances": importances,
    }


def _plain(value):
    """Convierte tipos de numpy a tipos serializables."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 6)
    return value

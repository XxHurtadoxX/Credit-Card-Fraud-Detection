"""Entrenamiento de punta a punta.

    python -m src.train

El orden de los pasos es lo que sostiene la validez del resultado. Primero se
separa el futuro y no se vuelve a tocar. Despues se comparan las funciones de
perdida sobre el pasado, por ahorro esperado. Con la ganadora se fija el umbral
en un tramo de validacion que tampoco es prueba. Al final, una sola vez, se mide
contra el futuro.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import date

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone

# Este modulo corre sin pantalla y solo guarda figuras a disco.
matplotlib.use("Agg")

from . import backtest, baselines, config, data, economics, evaluation, features, model

log = logging.getLogger("train")


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def run(n_iter: int = 20, skip_search: bool = False) -> dict:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = data.load_raw()

    log.info("Particion temporal")
    train_df, test_df = data.temporal_split(df)

    # Dentro del pasado, el ultimo tramo se reserva para elegir el umbral.
    cut = int(len(train_df) * (1 - config.VALIDATION_FRACTION))
    fit_df = train_df.iloc[:cut].reset_index(drop=True)
    val_df = train_df.iloc[cut:].reset_index(drop=True)
    log.info(
        "Ajuste %s filas / %s fraudes, validacion %s filas / %s fraudes",
        f"{len(fit_df):,}", int(fit_df[config.TARGET].sum()),
        f"{len(val_df):,}", int(val_df[config.TARGET].sum()),
    )

    X_train = features.build_features(train_df)
    y_train = train_df[config.TARGET].astype(int)
    X_fit, y_fit = features.build_features(fit_df), fit_df[config.TARGET].astype(int)
    X_val, y_val = features.build_features(val_df), val_df[config.TARGET].astype(int)
    X_test = features.build_features(test_df)
    y_test = test_df[config.TARGET].astype(int)

    amount_train = train_df["Amount"].astype(float)
    times_train = train_df["Time"].astype(float)

    log.info("Comparacion de perdidas por ahorro esperado")
    best_name, cv_results = model.select_candidate(
        X_train, y_train, amount_train, times_train,
    )

    base_pipeline = model.candidates(y_train)[best_name]

    if skip_search:
        log.info("Ajuste de hiperparametros omitido")
        tuned = clone(base_pipeline).fit(X_train, y_train)
        study = {"best_value": None, "best_params": {}, "n_trials": 0,
                 "param_importances": {}}
    else:
        log.info("Ajuste de hiperparametros, %s pruebas", n_iter)
        study = model.tune(
            base_pipeline, X_train, y_train, amount_train, times_train,
            n_trials=n_iter,
        )
        tuned = study.pop("estimator")

    # El umbral se fija con un modelo ajustado solo con el tramo previo.
    log.info("Fijando umbral sobre validacion")
    threshold_model = clone(tuned).fit(X_fit, y_fit)
    val_score = threshold_model.predict_proba(X_val)[:, 1]
    val_days = data.window_days(val_df["Time"])
    val_amount = val_df["Amount"].to_numpy()
    chosen = economics.pick_threshold(y_val, val_score, val_amount, val_days)
    log.info(
        "Umbral %.2f  recall %.3f  precision %.3f  %.1f alertas/dia",
        chosen.threshold, chosen.recall, chosen.precision, chosen.alerts_per_day,
    )

    # Reentrenamiento con todo el pasado y una sola medicion contra el futuro.
    log.info("Reentrenando con el tramo completo de entrenamiento")
    final = clone(tuned).fit(X_train, y_train)
    test_score = final.predict_proba(X_test)[:, 1]
    test_days = data.window_days(test_df["Time"])
    test_amount = test_df["Amount"].to_numpy()

    ranking = evaluation.ranking_metrics(y_test, test_score)
    at_precision = evaluation.recall_at_precision(y_test, test_score)
    test_cost = economics.cost_at_threshold(
        y_test, test_score, test_amount, chosen.threshold, test_days,
    )

    log.info(
        "Prueba: AP %.4f  recall %.3f  precision %.3f  ahorro %.0f de %.0f posible",
        ranking["average_precision"], test_cost.recall, test_cost.precision,
        test_cost.savings, test_cost.baseline_cost,
    )

    # Referencia: la regresion logistica sobre el mismo corte temporal.
    reference = clone(model.candidates(y_train)["logreg_balanced"]).fit(X_train, y_train)
    reference_score = reference.predict_proba(X_test)[:, 1]

    # Motor de reglas, calibrado sobre el mismo tramo de validacion que el
    # umbral del modelo y aplicado al mismo conjunto de prueba.
    log.info("Calibrando lineas base de reglas")
    tree = baselines.fit_rule_engine(X_fit, y_fit)
    rule_calibration = baselines.calibrate(
        X_val, y_val, val_amount, val_df["Time"], val_days, tree=tree,
    )
    rule_name, _ = baselines.best_rule(rule_calibration)

    rule_results = {}
    for name, calibrated in rule_calibration.items():
        scores = baselines.apply(
            name, calibrated["parametros"], X_test, test_amount,
            test_df["Time"], tree=tree,
        )
        outcome = economics.cost_at_threshold(
            y_test, scores, test_amount,
            calibrated["parametros"].get("threshold", 0.5), test_days,
        )
        rule_results[name] = {
            "descripcion": calibrated["descripcion"],
            "parametros": calibrated["parametros"],
            "test": outcome.as_dict(),
        }
        log.info("Regla %-11s ahorro en prueba %.1f%%  recall %.3f",
                 name, outcome.savings_share * 100, outcome.recall)

    rule_savings = rule_results[rule_name]["test"]["savings"]
    model_uplift = baselines.uplift(test_cost.savings, rule_savings)
    log.info("El modelo agrega %.1f puntos sobre la mejor regla (%s)",
             (test_cost.savings_share
              - rule_results[rule_name]["test"]["savings_share"]) * 100, rule_name)

    log.info("Generando figuras")
    evaluation.plot_pr_curves(
        {best_name: (y_test, test_score),
         "logreg_balanced": (y_test, reference_score)},
        ranking["prevalence"],
        config.FIGURES_DIR / "pr-curve.png",
    )
    curve = economics.cost_curve(y_test, test_score, test_amount, test_days)
    evaluation.plot_cost_curve(
        curve, chosen.threshold, config.DAILY_REVIEW_CAPACITY,
        config.FIGURES_DIR / "cost-vs-threshold.png",
        rules_cost=rule_results[rule_name]["test"]["total_cost"],
    )
    evaluation.plot_confusion(test_cost, config.FIGURES_DIR / "confusion.png")
    evaluation.plot_amount_by_class(df, config.FIGURES_DIR / "amount-by-class.png")

    importances = _gain_importances(final)
    if importances is not None:
        evaluation.plot_feature_importance(
            features.FEATURE_COLUMNS, importances,
            config.FIGURES_DIR / "feature-importance.png",
        )

    sensitivity = {
        "friction": economics.sensitivity(
            y_val, val_score, val_amount, val_days, "friction",
            [0, 10, 25, 50, 100],
        ),
        "review": economics.sensitivity(
            y_val, val_score, val_amount, val_days, "review",
            [0, 0.5, 1, 2, 4, 8],
        ),
        # Los dos barridos anteriores salen planos por separado. La rejilla
        # muestra que el umbral solo cede cuando ambos supuestos caen a la vez.
        "grid_review_friction": economics.sensitivity_grid(
            y_val, val_score, val_amount, val_days,
            {"review": [0, 1, 2, 4], "friction": [0, 2, 5, 10, 25]},
        ),
    }
    ceiling = economics.unreachable_fraud(y_test, test_score, test_amount)
    envelope = economics.operating_envelope(
        y_test, test_score, test_amount, test_days,
    )
    log.info("Techo de recall de valor con ahorro positivo: %.1f%%",
             envelope["max_value_recall_while_profitable"] * 100)

    # Backtest de ventana expansiva sobre toda la linea de tiempo. Reajusta y
    # recalibra en cada paso, de modo que muestra la dispersion entre periodos
    # que un promedio de validacion cruzada esconde.
    log.info("Backtest con ventana expansiva")
    backtest_summary = backtest.walk_forward(df, tuned, n_blocks=6)
    if backtest_summary["savings_share_mean"] is not None:
        log.info("Backtest: ahorro %.1f%% de media, entre %.1f%% y %.1f%% "
                 "en %s bloques",
                 backtest_summary["savings_share_mean"] * 100,
                 backtest_summary["savings_share_min"] * 100,
                 backtest_summary["savings_share_max"] * 100,
                 backtest_summary["n_blocks_evaluated"])

    payload = {
        "generated_on": date.today().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "dataset": {
            "rows": int(len(df)),
            "frauds": int(df[config.TARGET].sum()),
            "prevalence": round(float(df[config.TARGET].mean()), 6),
            "window_days": round(data.window_days(df["Time"]), 2),
        },
        "split": {
            "strategy": "temporal",
            "test_fraction": config.TEST_FRACTION,
            "train_rows": int(len(train_df)),
            "train_frauds": int(y_train.sum()),
            "validation_rows": int(len(val_df)),
            "validation_frauds": int(y_val.sum()),
            "test_rows": int(len(test_df)),
            "test_frauds": int(y_test.sum()),
            "test_window_days": round(test_days, 2),
        },
        "model_selection": {
            "criterion": "savings_share, umbral elegido dentro de cada fold",
            "candidates": cv_results,
            "selected": best_name,
            "hyperparameters": study,
        },
        "threshold": {
            "value": chosen.threshold,
            "chosen_on": "validation",
            "validation": chosen.as_dict(),
        },
        "test": {
            **ranking,
            **at_precision,
            "cost": test_cost.as_dict(),
            "unreachable_fraud": ceiling,
            "operating_envelope": envelope,
        },
        "baselines": {
            "note": ("Las reglas se calibran sobre el mismo tramo de validacion "
                     "que el umbral del modelo y se aplican al mismo conjunto de "
                     "prueba. El arbol es la vara relevante."),
            "rules": rule_results,
            "best_rule": rule_name,
            "rule_engine": baselines.rule_engine_text(tree, features.FEATURE_COLUMNS),
            "model_uplift_over_best_rule": model_uplift,
        },
        "backtest": backtest_summary,
        "reference_logreg": evaluation.ranking_metrics(y_test, reference_score),
        "cost_assumptions": {
            "review": config.COST_REVIEW,
            "friction": config.COST_FRICTION,
            "recovery_rate": config.RECOVERY_RATE,
            "chargeback_fee": config.CHARGEBACK_FEE,
            "daily_review_capacity": config.DAILY_REVIEW_CAPACITY,
        },
        "sensitivity": sensitivity,
    }

    evaluation.save_metrics(payload)
    log.info("Metricas en %s", config.METRICS_PATH)

    joblib.dump(
        {
            "pipeline": final,
            "threshold": chosen.threshold,
            "feature_columns": features.FEATURE_COLUMNS,
            "trained_on": date.today().isoformat(),
        },
        config.MODEL_PATH,
    )
    log.info("Modelo en %s", config.MODEL_PATH)

    # Los documentos se rearman desde las metricas recien escritas, de modo que
    # no puedan quedar afirmando cifras de una corrida anterior.
    from . import report
    report.render()

    return payload


def _gain_importances(pipeline):
    clf = pipeline.named_steps.get("clf")
    return getattr(clf, "feature_importances_", None)




def main():
    parser = argparse.ArgumentParser(description="Entrena el detector de fraude")
    parser.add_argument("--n-trials", type=int, default=40, dest="n_iter",
                        help="pruebas de ajuste de hiperparametros")
    parser.add_argument("--skip-search", action="store_true",
                        help="usa los hiperparametros por defecto")
    args = parser.parse_args()

    _setup_logging()
    payload = run(n_iter=args.n_iter, skip_search=args.skip_search)
    print(json.dumps(payload["test"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Backtesting con ventana expansiva y reajuste en cada paso.

Una validacion cruzada temporal reordena los folds y promedia. Un backtest
recorre la linea de tiempo una sola vez, y en cada paso hace lo que haria la
operacion real: reajusta el modelo con todo el pasado disponible, recalibra el
umbral con el tramo mas reciente de ese pasado, y decide sobre el bloque
siguiente sin volver a mirarlo.

La diferencia que importa no es el promedio, es la dispersion entre bloques. Un
modelo que ahorra 55 por ciento en promedio con bloques de 20 y 90 no es el mismo
producto que uno que ahorra 50 en todos.

Limitacion que conviene tener presente. El dataset cubre dos dias, asi que los
bloques son de horas y el numero de fraudes por bloque queda en decenas. El
procedimiento es el correcto y la evidencia que produce es indicativa, no
concluyente.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import baselines, config, data, economics, features

log = logging.getLogger(__name__)


def walk_forward(
    df: pd.DataFrame,
    pipeline,
    n_blocks: int = 6,
    initial_share: float = 0.45,
    calibration_share: float = 0.25,
    compare_rules: bool = True,
) -> dict:
    """Recorre la ventana hacia adelante y evalua bloque por bloque.

    `initial_share` es la fraccion del dataset que queda como historia antes del
    primer bloque de decision. El resto se parte en `n_blocks` tramos iguales.
    """
    n = len(df)
    start = int(n * initial_share)
    edges = np.linspace(start, n, n_blocks + 1).astype(int)

    rows = []
    for i in range(n_blocks):
        hist_end, block_end = edges[i], edges[i + 1]
        history = df.iloc[:hist_end]
        block = df.iloc[hist_end:block_end]

        if block.empty or history[config.TARGET].nunique() < 2:
            continue
        if block[config.TARGET].sum() == 0:
            log.info("Bloque %s sin fraudes, se omite", i + 1)
            continue

        # Dentro de la historia, el tramo final calibra el umbral.
        cut = int(len(history) * (1 - calibration_share))
        fit_part, cal_part = history.iloc[:cut], history.iloc[cut:]
        if cal_part[config.TARGET].sum() == 0 or fit_part[config.TARGET].nunique() < 2:
            continue

        X_fit = features.build_features(fit_part)
        y_fit = fit_part[config.TARGET].astype(int)
        X_cal = features.build_features(cal_part)
        y_cal = cal_part[config.TARGET].astype(int)
        X_block = features.build_features(block)
        y_block = block[config.TARGET].astype(int)

        amount_cal = cal_part["Amount"].to_numpy()
        amount_block = block["Amount"].to_numpy()
        days_cal = max(data.window_days(cal_part["Time"]), 1e-6)
        days_block = max(data.window_days(block["Time"]), 1e-6)

        model = clone(pipeline).fit(X_fit, y_fit)
        chosen = economics.pick_threshold(
            y_cal, model.predict_proba(X_cal)[:, 1], amount_cal, days_cal,
        )

        outcome = economics.cost_at_threshold(
            y_block, model.predict_proba(X_block)[:, 1],
            amount_block, chosen.threshold, days_block,
        )

        row = {
            "block": i + 1,
            "history_rows": int(len(history)),
            "block_rows": int(len(block)),
            "block_frauds": int(y_block.sum()),
            "block_days": round(days_block, 3),
            "threshold": chosen.threshold,
            "recall": outcome.recall,
            "precision": outcome.precision,
            "alerts_per_day": outcome.alerts_per_day,
            "savings": outcome.savings,
            "savings_share": outcome.savings_share,
        }

        if compare_rules:
            tree = baselines.fit_rule_engine(X_fit, y_fit)
            calib = baselines.calibrate(
                X_cal, y_cal, amount_cal, cal_part["Time"], days_cal, tree=tree,
            )
            rule_name, _ = baselines.best_rule(calib)
            rule_scores = baselines.apply(
                rule_name, calib[rule_name]["parametros"],
                X_block, amount_block, block["Time"], tree=tree,
            )
            rule_threshold = calib[rule_name]["parametros"].get("threshold", 0.5)
            rule_outcome = economics.cost_at_threshold(
                y_block, rule_scores, amount_block, rule_threshold, days_block,
            )
            row["rule"] = rule_name
            row["rule_savings_share"] = rule_outcome.savings_share
            row["uplift_points"] = round(
                (outcome.savings_share - rule_outcome.savings_share) * 100, 1)

        rows.append(row)
        log.info(
            "Bloque %s/%s  %s fraudes  umbral %.2f  ahorro %.1f%%%s",
            i + 1, n_blocks, row["block_frauds"], row["threshold"],
            row["savings_share"] * 100,
            f"  reglas {row['rule_savings_share'] * 100:.1f}%" if compare_rules else "",
        )

    shares = [r["savings_share"] for r in rows]
    summary = {
        "blocks": rows,
        "n_blocks_evaluated": len(rows),
        "savings_share_mean": round(float(np.mean(shares)), 4) if shares else None,
        "savings_share_std": round(float(np.std(shares)), 4) if shares else None,
        "savings_share_min": round(float(np.min(shares)), 4) if shares else None,
        "savings_share_max": round(float(np.max(shares)), 4) if shares else None,
    }
    if rows and "uplift_points" in rows[0]:
        ups = [r["uplift_points"] for r in rows]
        summary["uplift_points_mean"] = round(float(np.mean(ups)), 1)
        summary["uplift_points_min"] = round(float(np.min(ups)), 1)
        summary["blocks_where_model_wins"] = int(sum(u > 0 for u in ups))
    return summary

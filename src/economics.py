"""Traduccion de la matriz de confusion a dinero.

Un clasificador de fraude se elige por cuanto cuesta operarlo, y ese costo
depende del monto de cada transaccion. Dejar pasar un fraude de 3 unidades no se
parece en nada a dejar pasar uno de 2000, de modo que el costo se calcula
transaccion por transaccion en lugar de multiplicar conteos por un monto
promedio.

Los cuatro cuadrantes:

  fraude detectado      se recupera parte del monto, se paga una revision
  fraude no detectado   se pierde el monto completo mas el contracargo
  legitima bloqueada    se paga la revision y la friccion con el cliente
  legitima aprobada     no cuesta nada

El costo de friccion es el parametro mas incierto del conjunto, porque nadie
publica lo que cuesta molestar a un cliente. `sensitivity` y `sensitivity_grid`
muestran en que rango de supuestos la decision se mantiene.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import config


@dataclass
class CostBreakdown:
    """Desglose de costo para un umbral dado, en unidades monetarias."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    alerts: int
    alerts_per_day: float
    recall: float
    precision: float
    value_recall: float
    fraud_amount_total: float
    fraud_amount_recovered: float
    cost_missed_fraud: float
    cost_unrecovered_fraud: float
    cost_review: float
    cost_friction: float
    total_cost: float
    baseline_cost: float
    savings: float
    savings_share: float

    def as_dict(self) -> dict:
        return asdict(self)


def cost_at_threshold(
    y_true,
    y_score,
    amount,
    threshold: float,
    window_days: float,
    params: dict | None = None,
) -> CostBreakdown:
    """Evalua el costo operativo de aplicar `threshold` a `y_score`."""
    p = {
        "review": config.COST_REVIEW,
        "friction": config.COST_FRICTION,
        "recovery": config.RECOVERY_RATE,
        "chargeback": config.CHARGEBACK_FEE,
    }
    if params:
        p.update(params)

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    amount = np.asarray(amount, dtype=float)

    flagged = y_score >= threshold
    is_fraud = y_true == 1

    tp_mask = flagged & is_fraud
    fp_mask = flagged & ~is_fraud
    fn_mask = ~flagged & is_fraud

    tp, fp, fn = int(tp_mask.sum()), int(fp_mask.sum()), int(fn_mask.sum())
    tn = int((~flagged & ~is_fraud).sum())

    # Lo que se pierde de un fraude que pasa: el monto completo y el cargo fijo
    # por procesar el contracargo. Lo que se pierde de uno detectado: la parte
    # del monto que ya no se alcanza a recuperar.
    cost_missed = float(amount[fn_mask].sum() + p["chargeback"] * fn)
    cost_unrecovered = float(amount[tp_mask].sum() * (1 - p["recovery"]))
    cost_review = float(p["review"] * (tp + fp))
    cost_friction = float(p["friction"] * fp)

    total = cost_missed + cost_unrecovered + cost_review + cost_friction

    # Referencia: no hacer nada. Todos los fraudes se consuman y se paga el
    # contracargo de cada uno. Es el escenario que el modelo debe superar.
    fraud_total = float(amount[is_fraud].sum())
    baseline = fraud_total + p["chargeback"] * int(is_fraud.sum())

    savings = baseline - total

    return CostBreakdown(
        threshold=round(float(threshold), 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        alerts=tp + fp,
        alerts_per_day=round((tp + fp) / window_days, 1) if window_days else float("nan"),
        recall=round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        precision=round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        # Fraccion del dinero fraudulento que queda marcada. Es el recall que
        # importa cuando el costo depende del monto.
        value_recall=round(float(amount[tp_mask].sum() / fraud_total), 4)
        if fraud_total else 0.0,
        fraud_amount_total=round(fraud_total, 2),
        fraud_amount_recovered=round(float(amount[tp_mask].sum()) * p["recovery"], 2),
        cost_missed_fraud=round(cost_missed, 2),
        cost_unrecovered_fraud=round(cost_unrecovered, 2),
        cost_review=round(cost_review, 2),
        cost_friction=round(cost_friction, 2),
        total_cost=round(total, 2),
        baseline_cost=round(baseline, 2),
        savings=round(savings, 2),
        savings_share=round(savings / baseline, 4) if baseline else 0.0,
    )


def cost_curve(y_true, y_score, amount, window_days, thresholds=None, params=None):
    """Costo a lo largo de una rejilla de umbrales."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.99, 0.02), 3)
    return [
        cost_at_threshold(y_true, y_score, amount, t, window_days, params)
        for t in thresholds
    ]


def pick_threshold(
    y_true,
    y_score,
    amount,
    window_days,
    capacity_per_day: float | None = None,
    params: dict | None = None,
) -> CostBreakdown:
    """Elige el umbral de menor costo entre los operativamente viables.

    Se llama sobre validacion, nunca sobre prueba. Fijar el umbral con el
    conjunto de prueba y despues reportar el resultado en ese mismo conjunto
    sobreestima el desempeno.
    """
    capacity = config.DAILY_REVIEW_CAPACITY if capacity_per_day is None else capacity_per_day
    curve = cost_curve(y_true, y_score, amount, window_days, params=params)

    viable = [c for c in curve if c.alerts_per_day <= capacity]
    if not viable:
        # Ningun umbral cabe en la capacidad declarada. Se toma el que menos
        # alertas genera y el README debe decir que la capacidad es el limite.
        return min(curve, key=lambda c: c.alerts_per_day)
    return min(viable, key=lambda c: c.total_cost)


def sensitivity(y_true, y_score, amount, window_days, param, values, capacity=None):
    """Como se mueven umbral y ahorro cuando cambia un supuesto de costo.

    `param` es una de las claves de `cost_at_threshold`, es decir review,
    friction, recovery o chargeback. Un resultado plano indica que la decision no
    depende de ese supuesto en el rango probado.
    """
    rows = []
    for value in values:
        chosen = pick_threshold(
            y_true, y_score, amount, window_days,
            capacity_per_day=capacity, params={param: value},
        )
        rows.append({
            param: value,
            "threshold": chosen.threshold,
            "recall": chosen.recall,
            "precision": chosen.precision,
            "alerts_per_day": chosen.alerts_per_day,
            "savings": chosen.savings,
            "savings_share": chosen.savings_share,
        })
    return rows


def sensitivity_grid(y_true, y_score, amount, window_days, grid, capacity=None):
    """Umbral elegido variando dos supuestos a la vez.

    Un barrido de un parametro por vez puede salir plano porque el otro sostiene
    la decision, asi que la rejilla los mueve en conjunto.

    `grid` es un diccionario con dos claves y sus listas de valores.
    """
    (name_a, values_a), (name_b, values_b) = list(grid.items())
    rows = []
    for a in values_a:
        for b in values_b:
            chosen = pick_threshold(
                y_true, y_score, amount, window_days,
                capacity_per_day=capacity, params={name_a: a, name_b: b},
            )
            rows.append({
                name_a: a,
                name_b: b,
                "threshold": chosen.threshold,
                "recall": chosen.recall,
                "alerts_per_day": chosen.alerts_per_day,
                "savings": chosen.savings,
            })
    return rows


def top_k_outcome(y_true, y_score, amount, k: int, window_days: float) -> dict:
    """Resultado de revisar las k transacciones de mayor score.

    Una regla por presupuesto no tiene umbral que ajustar, asi que sirve para
    comparar modelos sin arrastrar el optimismo de elegir el corte y medir sobre
    los mismos datos.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    amount = np.asarray(amount, dtype=float)

    flagged = np.zeros(len(y_score), dtype=bool)
    flagged[np.argsort(-y_score, kind="stable")[:max(k, 0)]] = True

    tp = flagged & (y_true == 1)
    fraud_total = float(amount[y_true == 1].sum())
    breakdown = _from_masks(y_true, flagged, amount, window_days)

    return {
        "alerts": int(flagged.sum()),
        "alerts_per_day": round(flagged.sum() / window_days, 1) if window_days else None,
        "precision": round(float(tp.sum() / max(flagged.sum(), 1)), 4),
        "recall": round(float(tp.sum() / max((y_true == 1).sum(), 1)), 4),
        "value_recall": round(float(amount[tp].sum() / fraud_total), 4) if fraud_total else None,
        "savings": round(breakdown["savings"], 2),
        "savings_share": round(breakdown["savings_share"], 4),
    }


def _from_masks(y_true, flagged, amount, window_days, params=None) -> dict:
    """Costo a partir de una mascara de alertas ya decidida."""
    p = {
        "review": config.COST_REVIEW,
        "friction": config.COST_FRICTION,
        "recovery": config.RECOVERY_RATE,
        "chargeback": config.CHARGEBACK_FEE,
    }
    if params:
        p.update(params)

    is_fraud = y_true == 1
    tp, fp = flagged & is_fraud, flagged & ~is_fraud
    fn = ~flagged & is_fraud

    total = (amount[fn].sum() + p["chargeback"] * fn.sum()
             + amount[tp].sum() * (1 - p["recovery"])
             + p["review"] * (tp.sum() + fp.sum()) + p["friction"] * fp.sum())
    baseline = amount[is_fraud].sum() + p["chargeback"] * is_fraud.sum()
    return {"total_cost": float(total), "baseline_cost": float(baseline),
            "savings": float(baseline - total),
            "savings_share": float((baseline - total) / baseline) if baseline else 0.0}


def operating_envelope(y_true, y_score, amount, window_days) -> dict:
    """Que se puede exigir al modelo y a que precio.

    Tres restricciones que en una operacion real vienen dadas y no se negocian
    con el area de datos. La capacidad de la cola de revision, la precision
    minima que el equipo tolera antes de dejar de confiar en las alertas, y el
    piso de monto fraudulento que riesgo exige interceptar.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    amount = np.asarray(amount, dtype=float)
    fraud_total = float(amount[y_true == 1].sum())

    # Cada score observado es un corte posible, lo que evita que la resolucion
    # de una rejilla fija esconda el optimo.
    cuts = np.unique(y_score)
    if len(cuts) > 4000:
        cuts = cuts[:: len(cuts) // 4000]

    outcomes = []
    for cut in cuts:
        flagged = y_score >= cut
        if not flagged.any():
            continue
        tp = flagged & (y_true == 1)
        outcomes.append({
            "threshold": float(cut),
            "precision": float(tp.sum() / flagged.sum()),
            "recall": float(tp.sum() / max((y_true == 1).sum(), 1)),
            "value_recall": float(amount[tp].sum() / fraud_total) if fraud_total else 0.0,
            "alerts_per_day": float(flagged.sum() / window_days) if window_days else 0.0,
            **_from_masks(y_true, flagged, amount, window_days),
        })

    unconstrained = max(outcomes, key=lambda o: o["savings"])

    # El maximo absoluto de recall de valor es siempre 1, porque alertar sobre
    # todo intercepta todo el fraude. El numero que sirve es cuanto monto se
    # puede interceptar sin que la operacion deje de ahorrar dinero.
    profitable = [o for o in outcomes if o["savings"] > 0]
    ceiling = max((o["value_recall"] for o in profitable), default=0.0)

    def report(o):
        return {
            "threshold": round(o["threshold"], 4),
            "precision": round(o["precision"], 4),
            "recall": round(o["recall"], 4),
            "value_recall": round(o["value_recall"], 4),
            "alerts_per_day": round(o["alerts_per_day"], 1),
            "savings": round(o["savings"], 2),
            "savings_share": round(o["savings_share"], 4),
            "share_of_unconstrained": round(o["savings"] / unconstrained["savings"], 4)
            if unconstrained["savings"] else None,
        }

    by_capacity = {}
    for k_per_day in (25, 50, 100, 150, 200, 400, 800):
        k = max(int(round(k_per_day * window_days)), 1)
        result = top_k_outcome(y_true, y_score, amount, k, window_days)
        result["share_of_unconstrained"] = (
            round(result["savings"] / unconstrained["savings"], 4)
            if unconstrained["savings"] else None)
        by_capacity[str(k_per_day)] = result

    by_precision = {}
    for floor in (0.5, 0.75, 0.9, 0.95, 0.99):
        viable = [o for o in outcomes if o["precision"] >= floor]
        by_precision[str(floor)] = report(max(viable, key=lambda o: o["savings"])) if viable else None

    return {
        # La referencia es el mejor umbral fijo. Una regla por presupuesto puede
        # superarla, porque seleccionar las k de mayor score es mas expresivo
        # que cortar en un valor cuando los scores se agrupan.
        "reference_rule": "mejor umbral fijo",
        "unconstrained": report(unconstrained),
        "max_value_recall_while_profitable": round(ceiling, 4),
        "by_daily_capacity": by_capacity,
        "by_min_precision": by_precision,
    }


def unreachable_fraud(y_true, y_score, amount, score_floor: float = 0.01) -> dict:
    """Fraude que ningun umbral razonable alcanza a marcar.

    Marca el techo del proyecto. Cuando una parte del fraude recibe un score
    casi nulo, mover el umbral no la recupera y lo que falta esta en los
    predictores.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    amount = np.asarray(amount, dtype=float)

    is_fraud = y_true == 1
    invisible = is_fraud & (y_score < score_floor)
    fraud_total = float(amount[is_fraud].sum())

    return {
        "score_floor": score_floor,
        "frauds": int(is_fraud.sum()),
        "frauds_below_floor": int(invisible.sum()),
        "share_of_frauds": round(float(invisible.sum() / is_fraud.sum()), 4)
        if is_fraud.any() else None,
        "amount_below_floor": round(float(amount[invisible].sum()), 2),
        "share_of_fraud_amount": round(float(amount[invisible].sum() / fraud_total), 4)
        if fraud_total else None,
    }

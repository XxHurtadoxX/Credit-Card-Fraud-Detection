"""Perdida sensible al costo.

El costo de un error depende del monto de la transaccion, no solo de su clase.
Dejar pasar un fraude de 2.000 cuesta doscientas veces mas que dejar pasar uno
de 10, y una perdida que trate a los dos como el mismo evento entrena al modelo
para el problema equivocado.

La reduccion de clasificacion sensible al costo a clasificacion ponderada
(Elkan 2001, Zadrozny 2003) resuelve esto. A cada observacion se le asigna el
arrepentimiento de decidir mal sobre ella, y minimizar la log loss ponderada por
ese peso equivale a minimizar el costo esperado.

Los dos arrepentimientos:

    fraude no marcado   se pierde el monto y el contracargo en lugar de pagar
                        una revision y recuperar la mayor parte
                        w = r*A + F - c_r

    legitima marcada    se paga revision y friccion en lugar de nada
                        w = c_r + c_f

El peso del fraude crece con el monto, asi que el modelo dedica su capacidad a
las operaciones que mueven dinero. El de la legitima es constante porque su
monto no cambia lo que cuesta bloquearla.

Los pesos crudos resuelven la asimetria de costo pero no la de frecuencia. Con
un fraude cada 578 operaciones, el peso medio del fraude queda unas cuatro veces
el de la legitima, muy por debajo de lo que hace falta para que la clase
minoritaria pese en el gradiente. El esquema `balanced` reescala los pesos del
fraude para que el total por clase se iguale, y de ese modo conserva la
diferenciacion por monto dentro de los fraudes sin perder el balanceo entre
clases.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from . import config


def regret_weights(
    y,
    amount,
    params: dict | None = None,
    amount_cap: float | None = None,
    scheme: str = "linear",
) -> np.ndarray:
    """Peso de cada observacion segun el dinero que pone en juego.

    `amount_cap` recorta el monto antes de pesar. Sin recorte, una sola
    operacion muy grande puede dominar el gradiente de todo un fold.

    `scheme` controla como entra el monto y si se reequilibran las clases.

        linear      arrepentimiento crudo, lineal en el monto
        log         comprime la escala del monto
        balanced    lineal en el monto y reescalado para igualar el peso total
                    de cada clase
        count       ignora el monto y reproduce el balanceo por clase
    """
    p = {
        "review": config.COST_REVIEW,
        "friction": config.COST_FRICTION,
        "recovery": config.RECOVERY_RATE,
        "chargeback": config.CHARGEBACK_FEE,
    }
    if params:
        p.update(params)

    y = np.asarray(y).astype(int)
    amount = np.asarray(amount, dtype=float).clip(min=0.0)
    if amount_cap is not None:
        amount = np.minimum(amount, amount_cap)

    w_negative = p["review"] + p["friction"]

    if scheme == "count":
        positives = max(int((y == 1).sum()), 1)
        w_positive = np.full(len(y), w_negative * (y == 0).sum() / positives)
    elif scheme == "log":
        raw = p["recovery"] * amount + p["chargeback"] - p["review"]
        raw = np.clip(raw, 1e-6, None)
        # Se reescala el logaritmo para conservar el peso medio de la version
        # lineal, de modo que las dos variantes sean comparables.
        compressed = np.log1p(raw)
        mask = y == 1
        if mask.any() and compressed[mask].mean() > 0:
            compressed = compressed * (raw[mask].mean() / compressed[mask].mean())
        w_positive = compressed
    else:
        w_positive = np.clip(
            p["recovery"] * amount + p["chargeback"] - p["review"], 1e-6, None
        )

    weights = np.where(y == 1, w_positive, w_negative).astype(float)

    if scheme == "balanced":
        mask = y == 1
        n_pos, n_neg = int(mask.sum()), int((~mask).sum())
        if n_pos and weights[mask].sum() > 0:
            # Iguala el peso total de las dos clases sin aplanar las
            # diferencias de monto dentro de la clase fraude.
            target = w_negative * n_neg
            weights[mask] *= target / weights[mask].sum()

    return weights


class CostWeightedPipeline(BaseEstimator, ClassifierMixin):
    """Envuelve un pipeline completo y lo entrena con pesos de arrepentimiento.

    Envuelve el pipeline y no solo el clasificador porque el peso se calcula con
    el monto en su escala original. Un escalador intermedio devuelve un arreglo
    sin nombres de columna y con los valores ya transformados, de modo que desde
    dentro del pipeline el monto ya no es recuperable.

    Para sklearn esto es un unico estimador, asi que la validacion cruzada lo
    clona y le entrega subconjuntos de `X` sin transformar. Los pesos se
    recalculan en cada fold sobre las filas de ese fold, que es lo que evita el
    error silencioso de pasar `sample_weight` como parametro de ajuste.
    """

    def __init__(
        self,
        estimator=None,
        weight_step: str = "clf",
        amount_column: str = "Amount_log",
        scheme: str = "linear",
        amount_cap: float | None = 1000.0,
        cost_params: dict | None = None,
    ):
        self.estimator = estimator
        self.weight_step = weight_step
        self.amount_column = amount_column
        self.scheme = scheme
        self.amount_cap = amount_cap
        self.cost_params = cost_params

    def _amounts(self, X) -> np.ndarray:
        if not (hasattr(X, "columns") and self.amount_column in X.columns):
            raise ValueError(
                f"No se encontro la columna {self.amount_column!r}. "
                "CostWeightedPipeline necesita el monto sin transformar."
            )
        return np.expm1(np.asarray(X[self.amount_column], dtype=float))

    def fit(self, X, y, **fit_params):
        weights = regret_weights(
            y,
            self._amounts(X),
            params=self.cost_params,
            amount_cap=self.amount_cap,
            scheme=self.scheme,
        )
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(
            X, y, **{f"{self.weight_step}__sample_weight": weights}, **fit_params
        )
        self.classes_ = getattr(self.estimator_, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    @property
    def named_steps(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.named_steps

    @property
    def feature_importances_(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.named_steps[self.weight_step].feature_importances_

"""Metricas y figuras.

El ahorro esperado es lo que decide y estas metricas describen. Con 0.17 por
ciento de positivos, la exactitud de un modelo que niega todo fraude es 99.83 por
ciento, y el AUC ROC se mueve tan poco entre un modelo bueno y uno mediocre que
deja de discriminar. El area bajo la curva de precision y recall tiene como linea
base la prevalencia, asi que separa modelos y hace comparable el proyecto con la
literatura del dataset.

Tambien se reporta recall a precision fija, que responde cuantos fraudes se
alcanzan aceptando una proporcion dada de alertas falsas.
"""

from __future__ import annotations

import json

# Sin `matplotlib.use` aqui. Fijar el backend al importar este modulo apagaria
# las figuras en linea de cualquier notebook que lo use. Los scripts que corren
# sin pantalla eligen Agg por su cuenta; ver src/train.py.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from . import config

PALETTE = [config.COLOR_PATINA, config.COLOR_COPPER, config.COLOR_MID, "#2563EB"]


def apply_style():
    """Estilo comun de las figuras, alineado con la paleta de config."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": config.COLOR_LINE,
        "axes.labelcolor": config.COLOR_INK,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.grid": True,
        "grid.color": config.COLOR_LINE,
        "grid.linewidth": 0.8,
        "text.color": config.COLOR_INK,
        "xtick.color": config.COLOR_MID,
        "ytick.color": config.COLOR_MID,
        "font.size": 10,
        "legend.frameon": False,
        "figure.dpi": 130,
    })


def clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def ranking_metrics(y_true, y_score) -> dict:
    """Metricas que no dependen del umbral elegido."""
    y_true = np.asarray(y_true).astype(int)
    prevalence = float(y_true.mean())
    ap = float(average_precision_score(y_true, y_score))
    return {
        "average_precision": round(ap, 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "prevalence": round(prevalence, 6),
        # Cuantas veces mejor que ordenar al azar. Con una prevalencia de
        # 0.0017 el AP crudo necesita esta referencia para poder leerse.
        "ap_lift_over_baseline": round(ap / prevalence, 1) if prevalence else None,
    }


def recall_at_precision(y_true, y_score, targets=(0.25, 0.5, 0.75, 0.9)) -> dict:
    """Maximo recall alcanzable manteniendo una precision dada."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    out = {}
    for target in targets:
        ok = precision[:-1] >= target
        if ok.any():
            idx = int(np.argmax(recall[:-1] * ok))
            out[f"recall_at_precision_{int(target * 100)}"] = {
                "recall": round(float(recall[idx]), 4),
                "threshold": round(float(thresholds[idx]), 4),
            }
        else:
            out[f"recall_at_precision_{int(target * 100)}"] = None
    return out


def plot_pr_curves(curves: dict, prevalence: float, path):
    """Curvas de precision y recall superpuestas, con la linea base marcada."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for i, (label, (y_true, y_score)) in enumerate(curves.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=PALETTE[i % len(PALETTE)],
                linewidth=2 if i == 0 else 1.4,
                label=f"{label} (AP {ap:.3f})")

    ax.axhline(prevalence, color=config.COLOR_MID, linestyle=":", linewidth=1)
    ax.annotate(f"azar = {prevalence:.4f}", (0.55, prevalence),
                textcoords="offset points", xytext=(0, 6),
                fontsize=8, color=config.COLOR_MID)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision y recall en el conjunto de prueba")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cost_curve(curve, chosen_threshold: float, capacity: float, path,
                    rules_cost: float | None = None):
    """Costo total y alertas por dia frente al umbral.

    La referencia es el costo del motor de reglas, que es lo que la entidad gasta
    hoy. Aprobar todo aparece como techo teorico y no como alternativa, porque
    ninguna operacion real funciona sin control.
    """
    apply_style()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})

    thresholds = [c.threshold for c in curve]
    costs = [c.total_cost for c in curve]
    ax.plot(thresholds, costs, color=config.COLOR_PATINA, linewidth=2,
            label="Costo con el modelo")

    if rules_cost is not None:
        # La franja marca los umbrales donde el modelo cuesta menos que el
        # sistema actual, que es la region donde vale la pena desplegarlo.
        ax.fill_between(thresholds, costs, rules_cost,
                        where=[c <= rules_cost for c in costs],
                        color=config.COLOR_PATINA, alpha=0.10, linewidth=0)
        ax.axhline(rules_cost, color=config.COLOR_DANGER, linestyle="--",
                   linewidth=1.4, label="Costo del motor de reglas actual")

    ax.axhline(curve[0].baseline_cost, color=config.COLOR_MID, linestyle=":",
               linewidth=1, label="Sin ningun control")
    ax.axvline(chosen_threshold, color=config.COLOR_COPPER, linewidth=1.2)
    # Anclado en coordenadas de ejes para que no choque con la leyenda.
    ax.annotate(f"umbral elegido {chosen_threshold:g}",
                xy=(chosen_threshold, 0.06), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(-8, 0),
                ha="right", fontsize=8, color=config.COLOR_COPPER)
    ax.set_ylabel("Costo total de la ventana")
    ax.set_title("El umbral se elige por costo, no por estadistica")
    ax.legend(loc="center right", fontsize=8)
    clean_axes(ax)

    ax2.plot(thresholds, [c.alerts_per_day for c in curve],
             color=config.COLOR_MID, linewidth=1.6)
    ax2.axhline(capacity, color=config.COLOR_COPPER, linestyle=":", linewidth=1.2)
    ax2.annotate(f"capacidad {capacity:g}/dia",
                 xy=(0.5, capacity), xycoords=("axes fraction", "data"),
                 textcoords="offset points", xytext=(0, 5),
                 fontsize=8, color=config.COLOR_COPPER)
    ax2.axvline(chosen_threshold, color=config.COLOR_COPPER, linewidth=1.2)
    ax2.set_xlabel("Umbral de decision")
    ax2.set_ylabel("Alertas / dia")
    clean_axes(ax2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(breakdown, path):
    """Matriz de confusion anotada con lo que cuesta cada celda."""
    apply_style()
    matrix = np.array([
        [breakdown.true_negatives, breakdown.false_positives],
        [breakdown.false_negatives, breakdown.true_positives],
    ])
    labels = [
        ["Aprobadas correctamente", "Legitimas bloqueadas"],
        ["Fraudes no detectados", "Fraudes detectados"],
    ]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.imshow(np.log1p(matrix), cmap="Greys", alpha=0.28)
    ax.grid(False)

    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.08, f"{matrix[i, j]:,}", ha="center",
                    fontsize=17, fontweight="700", color=config.COLOR_INK)
            ax.text(j, i + 0.17, labels[i][j], ha="center", fontsize=8,
                    color=config.COLOR_MID)

    ax.set_xticks([0, 1], ["Aprobada", "Alertada"])
    ax.set_yticks([0, 1], ["Legitima", "Fraude"])
    ax.set_title(f"Decisiones en prueba, umbral {breakdown.threshold:g}")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_amount_by_class(df, path):
    """Distribucion del monto por clase, en escala logaritmica."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    for label, name, color in [(0, "Legitima", config.COLOR_MID),
                               (1, "Fraude", config.COLOR_COPPER)]:
        values = np.log1p(df.loc[df[config.TARGET] == label, "Amount"])
        ax.hist(values, bins=60, density=True, alpha=0.55 if label == 0 else 0.8,
                color=color, label=name)

    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Densidad")
    ax.set_title("El fraude no se distingue por el monto")
    ax.legend(fontsize=9)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(names, importances, path, top: int = 12):
    """Importancia por ganancia, recortada a las primeras columnas."""
    apply_style()
    order = np.argsort(importances)[::-1][:top]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    y = np.arange(len(order))[::-1]
    ax.barh(y, np.asarray(importances)[order], color=config.COLOR_PATINA, height=0.7)
    ax.set_yticks(y, [names[i] for i in order], fontsize=9)
    ax.set_xlabel("Ganancia relativa")
    ax.set_title(f"Columnas con mas peso en el modelo (top {top})")
    ax.grid(axis="y", visible=False)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_metrics(payload: dict, path=None):
    path = path or config.METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

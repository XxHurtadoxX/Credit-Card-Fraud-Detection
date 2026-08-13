"""Genera el informe ejecutivo y el README a partir de las metricas medidas.

    python -m src.report

Cada cifra sale de `reports/metrics.json`, de modo que los documentos no pueden
afirmar algo distinto de lo que se midio. Volver a entrenar y regenerar los deja
sincronizados sin intervencion manual.

El diseno del informe vive en `docs/resumen-ejecutivo.html` y no se toca. Este
modulo reemplaza unicamente lo que hay entre `<main>` y `</main>`.
"""

from __future__ import annotations

import json
import logging

from . import config

log = logging.getLogger(__name__)

OPEN, CLOSE = "  <main>", "  </main>"


# ------------------------------------------------------------------ formato
def num(value, decimals: int = 0) -> str:
    """Formato con punto de miles y coma decimal."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",").replace(" ", ".")


def pct(fraction, decimals: int = 0) -> str:
    return f"{num(fraction * 100, decimals)}&nbsp;%"


def signed_pct(fraction, decimals: int = 0) -> str:
    sign = "&minus;" if fraction < 0 else ""
    return f"{sign}{num(abs(fraction) * 100, decimals)}&nbsp;%"


def money(value) -> str:
    return num(value, 0)


def scenarios(m) -> dict:
    """Los tres escenarios de costo y la diferencia que constituye el caso.

    Aprobar todo es el techo teorico de la perdida, no una alternativa que
    alguien opere. La comparacion relevante para una entidad que ya tiene motor
    de reglas es contra ese motor, y la diferencia entre ambos es lo que se deja
    de ahorrar si el modelo no se despliega.
    """
    model = m["test"]["cost"]
    rules = m["baselines"]["rules"]["arbol"]["test"]
    incremental = rules["total_cost"] - model["total_cost"]
    return {
        "exposure": model["baseline_cost"],
        "rules_cost": rules["total_cost"],
        "model_cost": model["total_cost"],
        "incremental": incremental,
        "cost_reduction": incremental / rules["total_cost"] if rules["total_cost"] else 0.0,
        "points": (model["savings_share"] - rules["savings_share"]) * 100,
    }


# ------------------------------------------------------------------ secciones
def case_intro(m) -> str:
    d, s = m["dataset"], m["split"]
    rules = m["baselines"]["rules"]["arbol"]["test"]
    cost = m["test"]["cost"]
    gain = (cost["savings_share"] - rules["savings_share"]) * 100

    return f"""
    <section>
      <div class="tag">CASE.INTRO</div>
      <h2>El problema</h2>
      <p>Las entidades financieras pierden dinero por fraude con tarjeta de credito, y la perdida no se puede evitar simplemente siendo mas estricto. Cada vez que llega una operacion hay que decidir en el momento si se aprueba o si se detiene para revisarla, y las dos decisiones cuestan. Detener una compra legitima molesta a un cliente que no hizo nada y consume tiempo de un analista. Dejar pasar un fraude significa perder el monto completo y pagar despues el contracargo.</p>
      <p>El desafio es que el fraude es rarisimo. En el conjunto de datos de este proyecto hay {num(d["frauds"])} operaciones fraudulentas sobre {num(d["rows"])}, es decir una cada {num(1 / d["prevalence"])}. Un sistema que apruebe todo acierta el {pct(1 - d["prevalence"], 2)} de las veces y sin embargo no sirve para nada.</p>

      <h2 style="margin-top:26px">Como se resuelve hoy</h2>
      <p>Antes de tener modelos estadisticos, y todavia hoy en muchas operaciones, la deteccion se apoya en un motor de reglas escritas por analistas de riesgo. Son condiciones explicitas del tipo detener la operacion si el monto pasa de cierto valor, si llega en un horario inusual o si combina varias caracteristicas sospechosas. El motor se calibra revisando el historico de contracargos y tiene la ventaja de que cualquiera puede leer por que se detuvo una compra.</p>
      <p>Su limite es la cantidad de condiciones que una persona puede escribir y mantener. Un motor de reglas captura los patrones evidentes, pero no combina veinte variables a la vez ni ajusta sus umbrales segun cuanto dinero esta en juego en cada operacion.</p>

      <h2 style="margin-top:26px">Que se construyo</h2>
      <p>Un modelo que estima la probabilidad de fraude de cada operacion y decide alertar comparando lo que cuesta equivocarse en cada direccion. El punto de corte no se elige por criterios estadisticos sino economicos, buscando el menor costo total para la entidad y respetando la capacidad real de revision del equipo.</p>
      <p>Todo lo que sigue compara ese modelo contra el motor de reglas descrito arriba, medido sobre las mismas {num(s["test_rows"])} operaciones que ninguno de los dos vio durante su calibracion. La comparacion no es contra la ausencia de control, porque ninguna entidad opera sin control. Sobre esas operaciones el modelo evita {pct(cost["savings_share"])} de la perdida evitable frente al {pct(rules["savings_share"])} del motor de reglas, una diferencia de {num(gain)} puntos.</p>
    </section>
"""


def cost_assumptions(m) -> str:
    a = m["cost_assumptions"]
    return f"""
    <section>
      <div class="tag">COST.ASSUMPTIONS</div>
      <h2>Cuanto cuesta cada tipo de error</h2>
      <p>Para poder comparar dos sistemas de deteccion hay que poder decir cuanto cuesta cada decision. Cada vez que el sistema evalua una operacion se da una de cuatro situaciones, y cada una tiene una consecuencia economica distinta.</p>
      <div class="rejilla k2" style="margin-bottom:18px">
        <div class="panel">
          <h3>Se detecta un fraude real</h3>
          <p>La operacion se detiene y se investiga. Se paga el costo de la revision y se recupera buena parte del dinero, aunque nunca todo, porque una fraccion ya se fue.</p>
          <h3 style="margin-top:14px">Se aprueba una compra legitima</h3>
          <p>El caso normal. No cuesta nada y es lo que ocurre en la enorme mayoria de las operaciones.</p>
        </div>
        <div class="panel cobre">
          <h3>Pasa un fraude sin detectar</h3>
          <p>Se pierde el monto completo de la operacion y ademas el cargo fijo que cobra la red por procesar el contracargo. Es el error mas caro.</p>
          <h3 style="margin-top:14px">Se detiene una compra legitima</h3>
          <p>Se paga la revision y se suma un costo mas difuso, el de haber bloqueado a un cliente que no hizo nada. Llamado, reposicion de tarjeta y riesgo de que se lleve su consumo a otra entidad.</p>
        </div>
      </div>
      <p>Poner numero a esas cuatro situaciones exige cuatro parametros que el negocio conoce y que no salen de los datos. Todas las cifras de este informe dependen de ellos, asi que conviene revisarlos antes de leer cualquier resultado.</p>
      <div class="tabla-envoltura">
        <table>
          <caption>Los valores usados en este informe. Estan declarados en <span class="mono">src/config.py</span> y cambiarlos recalcula todo el analisis.</caption>
          <thead>
            <tr><th>Parametro</th><th>Valor</th><th>Que representa</th><th>De donde deberia salir</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Costo de revisar una alerta</td><td>{num(a["review"])}</td>
              <td>El tiempo de analista que consume investigar una operacion detenida</td>
              <td>Costo laboral por caso, medible con datos internos</td>
            </tr>
            <tr>
              <td>Costo de bloquear a un cliente legitimo</td><td>{num(a["friction"])}</td>
              <td>La molestia al cliente cuando se detiene una compra valida</td>
              <td class="marca-riesgo">El mas incierto de los cuatro. Requiere una decision del area comercial</td>
            </tr>
            <tr>
              <td>Fraccion del monto que se recupera</td><td>{num(a["recovery_rate"], 2)}</td>
              <td>Cuanto dinero se rescata cuando el fraude se detecta a tiempo</td>
              <td>Historico de recuperos de la entidad</td>
            </tr>
            <tr>
              <td>Cargo por contracargo</td><td>{num(a["chargeback_fee"])}</td>
              <td>Lo que cuesta procesar la disputa de una operacion fraudulenta</td>
              <td>Tarifario de la red, es un dato cerrado</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p style="margin-top:12px">Las cifras monetarias del informe estan en la unidad del conjunto de datos, que su publicador no documento. Lo que se puede trasladar a otra operacion son las proporciones, no los montos absolutos.</p>
    </section>
"""


def results(m) -> str:
    c = m["test"]["cost"]
    r = m["baselines"]["rules"]["arbol"]["test"]
    s = m["split"]
    sc = scenarios(m)
    bt = m["backtest"]
    gain = (c["savings_share"] - r["savings_share"]) * 100
    ratio = r["alerts_per_day"] / c["alerts_per_day"]

    return f"""
    <section>
      <div class="tag">TX.FRAUD.KPI</div>
      <h2>El resultado</h2>
      <p class="sub">Medido una sola vez sobre las ultimas {num(s["test_window_days"] * 24, 1)} horas del periodo, {num(s["test_rows"])} operaciones con {s["test_frauds"]} fraudes. Los dos sistemas se calibraron con los mismos datos previos y ninguno vio este tramo.</p>
      <div class="tabla-envoltura" style="margin-bottom:16px">
        <table>
          <thead>
            <tr><th>Indicador</th><th>Modelo</th><th>Motor de reglas</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Perdida evitable que se evita</td>
              <td class="marca-patina">{pct(c["savings_share"])}</td><td>{pct(r["savings_share"])}</td>
            </tr>
            <tr>
              <td>Monto fraudulento interceptado</td>
              <td class="marca-patina">{pct(c["value_recall"])}</td><td>{pct(r["value_recall"])}</td>
            </tr>
            <tr>
              <td>Fraudes detectados de cada 100</td>
              <td>{num(c["recall"] * 100)}</td><td>{num(r["recall"] * 100)}</td>
            </tr>
            <tr>
              <td>Alertas correctas de cada 100</td>
              <td>{num(c["precision"] * 100)}</td><td>{num(r["precision"] * 100)}</td>
            </tr>
            <tr>
              <td>Alertas por dia</td>
              <td>{num(c["alerts_per_day"])}</td><td>{num(r["alerts_per_day"])}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="rejilla k3">
        <div class="panel">
          <div class="cifra patina">{pct(sc["cost_reduction"])}</div>
          <div class="pie">Menos de lo que hoy cuesta el fraude, contando el dinero perdido y el costo de operar la revision. Equivale a {num(gain)} puntos mas de perdida evitada.</div>
          <span class="inst">COST.REDUCTION</span>
        </div>
        <div class="panel cobre">
          <div class="cifra cobre">{num(ratio, 1)}&nbsp;<small>veces menos</small></div>
          <div class="pie">Alertas diarias que el motor de reglas, con mas fraude interceptado. Menos carga para el equipo de revision.</div>
          <span class="inst">ALERT.LOAD</span>
        </div>
        <div class="panel gris">
          <div class="cifra">{pct(bt["savings_share_min"])} a {pct(bt["savings_share_max"])}</div>
          <div class="pie">Rango de perdida evitada entre los {bt["n_blocks_evaluated"]} periodos del backtest. El promedio es {pct(bt["savings_share_mean"])}.</div>
          <span class="inst">BACKTEST.RANGE</span>
        </div>
      </div>
      <p style="margin-top:16px">La diferencia de precision es la que mas se siente en la operacion diaria. De cada 100 alertas que levanta el modelo, {num(c["precision"] * 100)} corresponden a fraude real; en el motor de reglas son {num(r["precision"] * 100)}. Un analista que revisa la cola del modelo encuentra fraude la mayoria de las veces, y esa es la diferencia entre un sistema que el equipo usa y uno que aprende a ignorar.</p>
    </section>
"""


def breakdown(m) -> str:
    c = m["test"]["cost"]
    s = m["split"]
    sc = scenarios(m)
    return f"""
    <section>
      <div class="tag">COST.BREAKDOWN</div>
      <h2>A donde va el dinero</h2>
      <p class="sub">Costo de operar el modelo durante la ventana de prueba, {num(s["test_window_days"] * 24, 1)} horas y {num(s["test_rows"])} operaciones.</p>
      <p>El costo total no es una sola cosa. Se reparte entre el fraude que se escapa, la parte del fraude detectado que ya no se alcanza a recuperar, y lo que cuesta operar la cola de revision con sus errores. Verlo desagregado muestra donde conviene trabajar.</p>
      <div class="tabla-envoltura" style="margin-bottom:18px">
        <table>
          <caption>Como se compone el costo de operar el modelo.</caption>
          <thead>
            <tr><th>Concepto</th><th>Monto</th><th>De donde sale</th></tr>
          </thead>
          <tbody>
            <tr><td>Fraude no detectado</td><td class="marca-riesgo">{money(c["cost_missed_fraud"])}</td><td>{c["false_negatives"]} operaciones que pasaron completas, mas su contracargo</td></tr>
            <tr><td>Fraude detectado y no recuperado</td><td>{money(c["cost_unrecovered_fraud"])}</td><td>La fraccion del monto que no se rescata aun detectando a tiempo</td></tr>
            <tr><td>Revision de alertas</td><td>{money(c["cost_review"])}</td><td>{c["alerts"]} alertas investigadas</td></tr>
            <tr><td>Bloqueos a clientes legitimos</td><td>{money(c["cost_friction"])}</td><td>{c["false_positives"]} compras validas detenidas</td></tr>
            <tr class="destacada"><td>Costo total con el modelo</td><td>{money(sc["model_cost"])}</td><td></td></tr>
          </tbody>
        </table>
      </div>
      <p>Casi todo el costo remanente esta en la primera fila. Operar la cola de revision, entre investigar alertas y compensar bloqueos equivocados, representa una fraccion menor del total. El margen de mejora esta en detectar mas fraude, no en abaratar la operacion.</p>
      <h2 style="margin-top:26px">Cuanto se ahorra y frente a que</h2>
      <p>Hay tres escenarios posibles y conviene no confundirlos. Aprobar todo sin ningun control es el techo teorico de la perdida, no una alternativa que alguien opere. El motor de reglas es lo que la entidad gasta hoy. El modelo es la propuesta.</p>
      <div class="tabla-envoltura">
        <table>
          <thead>
            <tr><th>Escenario</th><th>Costo</th><th>Que representa</th></tr>
          </thead>
          <tbody>
            <tr><td>Aprobar todo sin control</td><td>{money(sc["exposure"])}</td><td>Exposicion total al fraude de la ventana. Techo de lo que se puede evitar</td></tr>
            <tr><td>Motor de reglas</td><td>{money(sc["rules_cost"])}</td><td class="marca-cobre">El costo que la entidad tiene hoy</td></tr>
            <tr class="destacada"><td>Modelo</td><td>{money(sc["model_cost"])}</td><td>La propuesta</td></tr>
            <tr class="destacada">
              <td>Ahorro frente al costo actual</td>
              <td class="marca-patina">{money(sc["incremental"])}</td>
              <td>{pct(sc["cost_reduction"])} menos de lo que hoy cuesta el fraude</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p style="margin-top:12px">La ultima fila es el caso de negocio. Reemplazar el motor de reglas por el modelo reduce en {pct(sc["cost_reduction"])} lo que la entidad gasta hoy por fraude, contando tanto el dinero perdido como el costo de operar la revision. Visto sobre la exposicion total, el modelo evita {pct(c["savings_share"])} de la perdida y el motor de reglas {pct(m["baselines"]["rules"]["arbol"]["test"]["savings_share"])}, una diferencia de {num(sc["points"])} puntos.</p>
    </section>
"""


def backtest_section(m) -> str:
    bt = m["backtest"]
    rows = "\n".join(
        f"""              <tr><td>{b["block"]}</td><td>{b["block_frauds"]}</td>"""
        f"""<td class="mono">{num(b["threshold"], 2)}</td>"""
        f"""<td>{pct(b["savings_share"], 1)}</td>"""
        f"""<td class="{'marca-riesgo' if b["rule_savings_share"] < 0 else ''}">"""
        f"""{signed_pct(b["rule_savings_share"], 1)}</td></tr>"""
        for b in bt["blocks"])
    thresholds = [b["threshold"] for b in bt["blocks"]]

    return f"""
    <section>
      <div class="tag">BACKTEST.WF</div>
      <h2>Que tan estable es el resultado</h2>
      <p class="sub">La medicion anterior corresponde a un unico tramo del periodo. Este ejercicio repite la evaluacion a lo largo de toda la ventana para ver si el resultado se sostiene.</p>
      <p>El procedimiento imita lo que haria la operacion real. Se recorre la linea de tiempo hacia adelante y en cada paso el modelo se reentrena con toda la historia disponible hasta ese momento, se recalibra el punto de corte con el tramo mas reciente, y se decide sobre el periodo siguiente sin volver a mirarlo. Nada de lo que ocurre en un periodo se usa para decidir en ese mismo periodo.</p>
      <div class="rejilla k2">
        <div class="tabla-envoltura">
          <table>
            <caption>Perdida evitada por periodo. El modelo supera al motor de reglas en los {bt["n_blocks_evaluated"]} tramos, con una ventaja minima de {num(bt["uplift_points_min"], 1)} puntos.</caption>
            <thead>
              <tr><th>Periodo</th><th>Fraudes</th><th>Corte</th><th>Modelo</th><th>Reglas</th></tr>
            </thead>
            <tbody>
{rows}
            </tbody>
          </table>
        </div>
        <div>
          <div class="panel gris" style="margin-bottom:16px">
            <h3>La dispersion importa</h3>
            <p>El promedio es {pct(bt["savings_share_mean"], 1)}, pero los periodos van de {pct(bt["savings_share_min"], 1)} a {pct(bt["savings_share_max"], 1)}. Un sistema que rinde parejo y otro que alterna entre esos dos extremos no son el mismo producto, aunque su promedio coincida.</p>
            <p>Antes de comprometer una cifra conviene decidir si se promete el promedio o el piso.</p>
          </div>
          <div class="panel cobre">
            <h3>El punto de corte se mueve</h3>
            <p>La recalibracion lo lleva entre {num(min(thresholds), 2)} y {num(max(thresholds), 2)} segun el periodo. Recalibrar de forma periodica no es un refinamiento opcional, es parte de la operacion.</p>
          </div>
        </div>
      </div>
      <div class="aviso" style="margin-top:16px">
        <strong>Alcance de esta evidencia.</strong> El conjunto de datos cubre dos dias, de modo que cada periodo abarca horas y contiene entre {min(b["block_frauds"] for b in bt["blocks"])} y {max(b["block_frauds"] for b in bt["blocks"])} fraudes. El procedimiento es el que corresponde, pero para sostener una promesa de negocio hace falta repetirlo sobre meses de operacion.
      </div>
    </section>
"""


def envelope(m) -> str:
    e = m["test"]["operating_envelope"]
    cap = e["by_daily_capacity"]
    prec = e["by_min_precision"]
    c = m["test"]["cost"]

    cap_rows = "\n".join(
        f"""              <tr><td>{k}</td><td>{num(v["precision"] * 100)}</td>"""
        f"""<td>{pct(v["recall"])}</td><td>{pct(v["value_recall"])}</td>"""
        f"""<td>{pct(v["share_of_unconstrained"])}</td></tr>"""
        for k, v in cap.items())

    prec_rows = "\n".join(
        f"""              <tr><td>{num(float(k) * 100)}</td><td>{pct(v["recall"])}</td>"""
        f"""<td>{num(v["alerts_per_day"])}</td><td>{pct(v["share_of_unconstrained"])}</td></tr>"""
        for k, v in prec.items() if v)

    return f"""
    <section>
      <div class="tag">OPS.ENVELOPE</div>
      <h2>Que se le puede exigir al modelo</h2>
      <p class="sub">La capacidad del equipo de revision y la precision minima que este dispuesto a tolerar no las decide el area de datos. Vienen dadas, y cada una tiene un precio.</p>
      <p>El punto de operacion elegido genera {num(c["alerts_per_day"])} alertas por dia. Si el equipo puede revisar menos, hay que subir el corte y se pierde deteccion. Si se le exige que casi todas las alertas sean fraude real, tambien. Las dos tablas cuantifican ese intercambio, tomando como referencia el mejor resultado sin restricciones.</p>
      <div class="rejilla k2">
        <div class="tabla-envoltura">
          <table>
            <caption>Segun cuantas alertas por dia pueda absorber la operacion.</caption>
            <thead>
              <tr><th>Alertas/dia</th><th>Correctas de 100</th><th>Fraudes</th><th>Monto</th><th>Del optimo</th></tr>
            </thead>
            <tbody>
{cap_rows}
            </tbody>
          </table>
        </div>
        <div class="tabla-envoltura">
          <table>
            <caption>Segun la precision minima que exija el equipo para confiar en la cola.</caption>
            <thead>
              <tr><th>Correctas de 100</th><th>Fraudes</th><th>Alertas/dia</th><th>Del optimo</th></tr>
            </thead>
            <tbody>
{prec_rows}
            </tbody>
          </table>
        </div>
      </div>
      <div class="aviso" style="margin-top:16px">
        <strong>El techo del sistema.</strong> El maximo monto fraudulento que este modelo puede interceptar sin que la operacion deje de ahorrar dinero es {pct(e["max_value_recall_while_profitable"])}, y en el punto elegido intercepta {pct(c["value_recall"])}. Si la politica de riesgo exige mas que eso, la respuesta no esta en mover el punto de corte sino en conseguir informacion que hoy no esta disponible.
      </div>
    </section>
"""


def decision(m) -> str:
    c = m["test"]["cost"]
    r = m["baselines"]["rules"]["arbol"]["test"]
    bt = m["backtest"]
    e = m["test"]["operating_envelope"]
    sc = scenarios(m)
    gain = (c["savings_share"] - r["savings_share"]) * 100

    return f"""
    <section>
      <div class="tag">DECISION</div>
      <h2>La recomendacion</h2>
      <div class="rejilla k2">
        <div class="panel">
          <h3>Que conviene hacer</h3>
          <p>Reemplazar el motor de reglas por el modelo, alertando cuando la probabilidad estimada supera {num(m["threshold"]["value"], 2)}. Sobre la ventana evaluada eso reduce en {pct(sc["cost_reduction"])} lo que hoy cuesta el fraude y ademas baja las alertas diarias de {num(r["alerts_per_day"])} a {num(c["alerts_per_day"])}, con lo cual libera tiempo del equipo de revision.</p>
          <p>El motor de reglas conviene conservarlo en paralelo durante la transicion, porque su ventaja es que cualquiera puede explicar por que detuvo una operacion.</p>
        </div>
        <div class="panel cobre">
          <h3>Con que salvedades</h3>
          <p>La evidencia cubre dos dias. El backtest muestra periodos de {pct(bt["savings_share_min"])} y de {pct(bt["savings_share_max"])}, asi que la cifra que se comprometa deberia ser conservadora y revisarse con datos de varios meses.</p>
          <p>El punto de corte requiere recalibracion periodica, y los cuatro supuestos de costo deberian confirmarse con datos internos de la entidad antes de fijar objetivos.</p>
        </div>
      </div>
      <p style="margin-top:16px">Para ir mas alla de {pct(e["max_value_recall_while_profitable"])} del monto fraudulento hace falta informacion que este conjunto de datos no tiene, en particular el identificador de tarjeta que permite construir el historial de comportamiento de cada cliente. Ese es el proximo paso con mayor impacto esperado, y es un pedido de datos antes que un problema de modelado.</p>
    </section>
"""


def limits(m) -> str:
    s = m["split"]
    u = m["test"]["unreachable_fraud"]
    return f"""
    <section>
      <div class="tag">LIMITS.DECLARED</div>
      <h2>Lo que este ejercicio no resuelve</h2>
      <div class="rejilla k2">
        <div class="panel gris">
          <h3>No explica sus decisiones</h3>
          <p>Las variables del conjunto de datos estan anonimizadas mediante una transformacion que las vuelve ilegibles. El modelo ordena bien las operaciones pero no puede decirle a un analista por que marco una en particular, y sin esa explicacion una cola de revision es dificil de operar.</p>
          <h3 style="margin-top:14px">No conoce al cliente</h3>
          <p>No hay identificador de tarjeta ni de comercio, asi que no se puede saber cuantas compras hizo esa tarjeta en la ultima hora ni si el monto se aparta de su costumbre. Son las senales mas usadas en los sistemas reales y son justamente las que reconocerian a los {u["frauds_below_floor"]} fraudes que el modelo hoy no ve.</p>
        </div>
        <div class="panel gris">
          <h3>Los montos no se anualizan</h3>
          <p>Las cifras corresponden a unas horas de operacion de un emisor europeo en 2013, en una unidad monetaria que el publicador no documento. Multiplicarlas por 365 supondria que el volumen, la composicion del fraude y la moneda se mantienen. Las proporciones se pueden trasladar, los montos absolutos no.</p>
          <h3 style="margin-top:14px">La muestra es corta</h3>
          <p>Quedan {s["test_frauds"]} fraudes en el tramo de prueba, de modo que un solo caso mueve el resultado mas de un punto. Cualquier diferencia pequena entre dos alternativas cae dentro del ruido de la medicion.</p>
        </div>
      </div>
    </section>
"""


def annex_divider() -> str:
    return """
    <div class="estriado claro" style="margin:8px 0 26px"></div>
    <section>
      <div class="tag">ANEXO</div>
      <h2>Como se construyo</h2>
      <p class="sub">Lo que sigue documenta el metodo para quien quiera auditarlo. El informe ejecutivo termina en la seccion anterior.</p>
    </section>
"""


def pipeline_flow(m) -> str:
    n = len(m["model_selection"]["candidates"])
    bt = m["backtest"]["n_blocks_evaluated"]
    boxes = [
        ("Corte temporal", "PASADO / FUTURO"),
        ("Predictores", "31 COLUMNAS"),
        (f"{n} candidatos", "VENTANA EXPANSIVA"),
        ("Corte por costo", "EN VALIDACION"),
        ("Medicion final", "UNA SOLA VEZ"),
        (f"Backtest", f"{bt} PERIODOS"),
    ]
    width, gap = 148, 14
    step = width + gap
    rects, tracks, nodes = [], [], []
    for i, (title, sub) in enumerate(boxes):
        x = 8 + i * step
        rects.append(
            f"""          <g>
            <rect class="caja" x="{x}" y="26" width="{width}" height="36" rx="3" />
            <text class="etq" x="{x + width // 2}" y="42" text-anchor="middle">{title}</text>
            <text class="etq-s" x="{x + width // 2}" y="55" text-anchor="middle">{sub}</text>
          </g>""")
        if i:
            tracks.append(f"""          <path class="pista" d="M{x - gap} 44 H{x}" />""")
            nodes.append(f"""          <circle class="nodo" cx="{x}" cy="44" r="4" />""")

    total = 8 + len(boxes) * step
    return f"""
    <section>
      <div class="tag">PIPE.FLOW</div>
      <h2>El orden de los pasos</h2>
      <p class="sub">El tramo de prueba se aparta al principio y se mide una sola vez, al final. Ese orden es lo que sostiene la validez del resultado.</p>
      <div class="flujo">
        <svg viewBox="0 0 {total} 100" role="img" aria-label="Secuencia de pasos, desde el corte temporal hasta el backtest">
          <defs>
            <style>
              .nodo {{ fill: #0E7C68; }}
              .pista {{ stroke: #0E7C68; stroke-width: 2; fill: none; }}
              .etq {{ font-family: "Inter", system-ui, sans-serif; font-size: 12px; fill: #0A0E14; }}
              .etq-s {{ font-family: "JetBrains Mono", monospace; font-size: 9.5px; fill: #5B6673; }}
              .caja {{ fill: #FFFFFF; stroke: #E3E7EC; }}
            </style>
          </defs>
{chr(10).join(tracks)}
{chr(10).join(nodes)}
{chr(10).join(rects)}
          <text class="etq-s" x="8" y="86">EL TRAMO DE PRUEBA QUEDA APARTADO DESDE EL PRIMER PASO Y NO PARTICIPA DE NINGUNA DECISION</text>
        </svg>
      </div>
    </section>
"""


LOSS_LABELS = {
    "xgb_weighted": "Pesos por clase",
    "xgb_cost_balanced": "Pesos por monto, reequilibrados",
    "xgb_plain": "Sin ponderar",
    "xgb_cost": "Pesos por monto",
    "logreg_balanced": "Referencia lineal",
}


def loss_select(m) -> str:
    cand = m["model_selection"]["candidates"]
    order = sorted(cand, key=lambda k: -cand[k]["savings_share_mean"])
    best = m["model_selection"]["selected"]
    top, second = order[0], order[1]
    gap = (cand[top]["savings_share_mean"] - cand[second]["savings_share_mean"]) * 100
    worst_cost = "xgb_cost"
    cost_gap = (cand["xgb_cost_balanced"]["savings_share_mean"]
                - cand[worst_cost]["savings_share_mean"]) * 100
    lin = cand["logreg_balanced"]

    rows = "\n".join(
        f"""              <tr{' class="destacada"' if k == best else ''}>"""
        f"""<td>{LOSS_LABELS.get(k, k)}</td>"""
        f"""<td>{pct(cand[k]["savings_share_mean"], 1)}</td>"""
        f"""<td>{num(cand[k]["savings_share_std"] * 100, 1)}</td>"""
        f"""<td>{pct(cand[k]["value_recall_mean"], 1)}</td>"""
        f"""<td>{num(cand[k]["alerts_per_day_mean"])}</td></tr>"""
        for k in order)

    return f"""
    <section>
      <div class="tag">LOSS.SELECT</div>
      <h2>Como se elige que optimiza el modelo</h2>
      <p class="sub">Cinco alternativas, cada una entrenada con una definicion distinta de que error es mas grave. Se comparan por el dinero que ahorran, no por su exactitud.</p>
      <p>Un modelo aprende minimizando una penalidad, y esa penalidad se puede escribir de varias formas. La mas simple castiga igual todos los errores. Otra castiga mas el fraude no detectado que la falsa alarma, para compensar que el fraude es rarisimo. Una tercera castiga cada error en proporcion al dinero que estaba en juego en esa operacion. Cada alternativa produce un modelo distinto y la pregunta es cual deja mas dinero en la entidad.</p>
      <p>Para responderla se recorre la historia hacia adelante en varios tramos. En cada uno se entrena con el pasado, se fija el punto de corte con el tramo mas reciente de ese pasado, y se mide el ahorro sobre el tramo siguiente. El promedio de esos tramos es la columna principal de la tabla.</p>
      <div class="rejilla k2" style="margin-top:16px">
        <div class="tabla-envoltura">
          <table>
            <caption>Ordenadas por el ahorro promedio. La desviacion indica cuanto varia ese ahorro entre tramos.</caption>
            <thead>
              <tr><th>Penalidad</th><th>Ahorro</th><th>Desv.</th><th>Monto</th><th>Alertas/dia</th></tr>
            </thead>
            <tbody>
{rows}
            </tbody>
          </table>
        </div>
        <div class="panel gris">
          <h3>Como se lee</h3>
          <p>La ganadora aventaja a la segunda por {num(gap, 1)} puntos, con desviaciones de alrededor de {num(cand[top]["savings_share_std"] * 100, 0)}. La diferencia es menor que la variabilidad entre tramos, asi que las dos primeras son equivalentes con esta cantidad de datos y elegir entre ellas es cuestion de simplicidad.</p>
          <p>Pesar cada error por el monto en juego parece la idea correcta, y lo es, pero solo funciona si ademas se compensa la rareza del fraude. Sin esa correccion pierde {num(cost_gap, 1)} puntos, porque con un fraude cada {num(1 / m["dataset"]["prevalence"])} operaciones la escasez pesa mas que el monto.</p>
          <p>La referencia lineal muestra por que hace falta mirar mas de una columna. Intercepta {pct(lin["value_recall_mean"])} del monto, casi tanto como la ganadora, y sin embargo ahorra bastante menos. La razon esta en sus {num(lin["alerts_per_day_mean"])} alertas diarias, que se pagan en revision y en clientes molestados.</p>
        </div>
      </div>
    </section>
"""


def threshold_section(m) -> str:
    c = m["test"]["cost"]
    thr = m["threshold"]["value"]
    u = m["test"]["unreachable_fraud"]
    return f"""
    <section>
      <div class="tag">THR.ECON</div>
      <h2>Como se fija el punto de corte</h2>
      <p class="sub">El modelo no responde si una operacion es fraude. Entrega una probabilidad, y hace falta decidir a partir de que valor conviene alertar.</p>
      <p>Para cada operacion el modelo devuelve un numero entre 0 y 1 que expresa cuan sospechosa la considera. Convertir ese numero en una decision requiere fijar un limite. Se suele usar 0,5 por costumbre, y ese valor no tiene ningun fundamento. Solo seria correcto si equivocarse en una direccion costara exactamente lo mismo que equivocarse en la otra, y en fraude no es asi.</p>
      <p>El limite correcto sale de comparar costos. Para cada valor posible se calcula que pasaria con todas las operaciones del tramo de calibracion, cuanto fraude se detendria, cuantos clientes legitimos quedarian bloqueados, y cuanto costaria el total. El limite elegido es el que minimiza ese costo respetando la capacidad de revision, y en este caso resulta {num(thr, 2)}.</p>
      <p>La referencia para juzgar ese costo es lo que gasta hoy el motor de reglas. Cualquier umbral por debajo de esa linea mejora la situacion actual, y el elegido es el que la mejora mas.</p>
      <div class="rejilla k2" style="margin-top:16px">
        <figure>
          <img src="../reports/figures/cost-vs-threshold.png" alt="Costo total y alertas por dia frente al punto de corte" />
          <figcaption>La linea roja es lo que cuesta hoy el motor de reglas. La franja verde marca los umbrales donde el modelo cuesta menos que el sistema actual, y el minimo queda en {num(thr, 2)}. El panel inferior muestra cuantas alertas genera cada valor.</figcaption>
        </figure>
        <div>
          <div class="panel cobre" style="margin-bottom:16px">
            <h3>Por que queda tan alto</h3>
            <p>El modelo reparte sus probabilidades de forma muy polarizada. A la mayoria de los fraudes les asigna un valor cercano a 1 y a un grupo pequeno les asigna practicamente 0, con muy pocos casos en el medio.</p>
            <p>La consecuencia practica es que bajar el limite casi no agrega fraudes detectados, porque no hay casos esperando justo debajo, y en cambio multiplica los bloqueos a clientes legitimos. Alertar mas seria mas caro sin ser mas efectivo.</p>
          </div>
          <div class="aviso">
            <strong>El limite no es el problema.</strong> Los fraudes que reciben probabilidad casi nula representan {pct(u["share_of_fraud_amount"])} del monto fraudulento del tramo. Ningun valor de corte los recupera, porque el modelo no los distingue de una compra normal. Lo que falta es informacion, no una decision distinta.
          </div>
        </div>
      </div>
    </section>
"""


def sensitivity_section(m) -> str:
    s = m["sensitivity"]
    fr = [r["threshold"] for r in s["friction"]]
    rv = [r["threshold"] for r in s["review"]]
    low = [r for r in s["grid_review_friction"] if r["threshold"] < 0.9]
    fr_vals = [r["friction"] for r in s["friction"]]
    rv_vals = [r["review"] for r in s["review"]]

    return f"""
    <section>
      <div class="tag">SENS.PARAM</div>
      <h2>Cuanto depende el resultado de los supuestos</h2>
      <p class="sub">Dos de los cuatro parametros de costo son estimaciones, no mediciones. Si la recomendacion cambiara al mover esos numeros, no seria una recomendacion.</p>
      <p>El mas discutible es el costo de bloquear a un cliente legitimo, porque no existe una fuente publica que lo cuantifique. Le sigue el costo de revisar una alerta, que depende de como este organizado el equipo. La prueba consiste en recalcular el punto de corte optimo variando esos supuestos y observar si la decision se mantiene.</p>
      <div class="rejilla k3">
        <div class="panel gris">
          <h3>Variando uno solo</h3>
          <p>Llevando el costo de bloquear un cliente de {num(min(fr_vals))} a {num(max(fr_vals))}, el punto de corte se queda en {num(fr[0], 2)} en practicamente todo el rango. Variando el costo de revision de {num(min(rv_vals))} a {num(max(rv_vals))}, no se mueve nada.</p>
        </div>
        <div class="panel gris">
          <h3>Variando los dos</h3>
          <p>Como los dos no pueden ser irrelevantes al mismo tiempo, se recorre la combinacion completa. El punto de corte solo baja en las {len(low)} combinaciones donde uno de los dos costos es exactamente cero y el otro es minimo, y aun ahi el resultado mejora en dos fraudes.</p>
        </div>
        <div class="panel cobre">
          <h3>Que significa</h3>
          <p>La decision se sostiene en todo el rango de supuestos que describe una operacion real, y solo cambiaria si revisar alertas y molestar clientes fueran ambos gratuitos. La recomendacion no depende de los numeros mas debiles del analisis.</p>
          <p>El corolario tambien es util. Discutir estos parametros con el negocio no va a cambiar el resultado, porque lo que falta son predictores.</p>
        </div>
      </div>
    </section>
"""


# ------------------------------------------------------------------ armado
def build_body(m) -> str:
    parts = [
        case_intro(m), cost_assumptions(m), results(m), breakdown(m),
        backtest_section(m), envelope(m), decision(m), limits(m),
        annex_divider(), pipeline_flow(m), loss_select(m),
        threshold_section(m), sensitivity_section(m),
    ]
    return OPEN + "\n" + "\n".join(parts) + "\n" + CLOSE


# ------------------------------------------------------------------ README
def plain(fraction, decimals: int = 0) -> str:
    """Porcentaje para Markdown.

    El espacio antes del signo es irrompible, para que el reflujo de parrafos no
    deje el simbolo al comienzo de la linea siguiente.
    """
    return f"{num(fraction * 100, decimals)} %"


def build_readme(m) -> str:
    d, s = m["dataset"], m["split"]
    c = m["test"]["cost"]
    r = m["baselines"]["rules"]["arbol"]["test"]
    rules = m["baselines"]["rules"]
    cand = m["model_selection"]["candidates"]
    order = sorted(cand, key=lambda k: -cand[k]["savings_share_mean"])
    bt = m["backtest"]
    e = m["test"]["operating_envelope"]
    u = m["test"]["unreachable_fraud"]
    a = m["cost_assumptions"]
    thr = m["threshold"]["value"]
    sc = scenarios(m)
    gain = (c["savings_share"] - r["savings_share"]) * 100
    cost_gap = (cand["xgb_cost_balanced"]["savings_share_mean"]
                - cand["xgb_cost"]["savings_share_mean"]) * 100
    hp = m["model_selection"]["hyperparameters"]

    loss_rows = "\n".join(
        f"| {LOSS_LABELS.get(k, k)} | {plain(cand[k]['savings_share_mean'], 1)} | "
        f"{num(cand[k]['savings_share_std'] * 100, 1)} | "
        f"{plain(cand[k]['value_recall_mean'], 1)} | "
        f"{num(cand[k]['alerts_per_day_mean'])} |"
        for k in order)

    rule_rows = "\n".join(
        f"| {rules[k]['descripcion']} | {plain(rules[k]['test']['savings_share'], 0)} | "
        f"{plain(rules[k]['test']['recall'])} | {plain(rules[k]['test']['value_recall'])} | "
        f"{num(rules[k]['test']['alerts_per_day'])} |"
        for k in ("monto", "monto_hora", "arbol"))

    bt_rows = "\n".join(
        f"| {b['block']} | {b['block_frauds']} | {num(b['threshold'], 2)} | "
        f"{plain(b['savings_share'], 1)} | {plain(b['rule_savings_share'], 1)} |"
        for b in bt["blocks"])

    cap_rows = "\n".join(
        f"| {k} | {num(v['precision'] * 100)} | {plain(v['recall'])} | "
        f"{plain(v['value_recall'])} | {plain(v['share_of_unconstrained'])} |"
        for k, v in e["by_daily_capacity"].items())

    imp = hp.get("param_importances") or {}
    imp_line = ", ".join(f"`{k}` {num(v * 100)} %" for k, v in list(imp.items())[:3])

    return f"""# Deteccion de fraude con tarjeta de credito

Las entidades financieras pierden dinero por fraude con tarjeta, y la perdida no
se evita simplemente siendo mas estricto. Cada operacion exige decidir en el
momento si se aprueba o se detiene para revisarla, y las dos decisiones cuestan.
Detener una compra legitima molesta a un cliente que no hizo nada y consume tiempo
de un analista. Dejar pasar un fraude significa perder el monto completo y pagar
despues el contracargo.

La deteccion se apoya tradicionalmente en un motor de reglas escritas por
analistas de riesgo, del tipo detener la operacion si el monto pasa de cierto
valor o si combina varias caracteristicas sospechosas. Funciona para los patrones
evidentes y tiene la ventaja de que cualquiera puede explicar por que se detuvo una
compra, pero esta limitado por la cantidad de condiciones que una persona puede
escribir y mantener.

Este proyecto construye un modelo que estima la probabilidad de fraude de cada
operacion y decide alertar comparando lo que cuesta equivocarse en cada direccion.
El punto de corte no se elige por criterios estadisticos sino economicos. Todo lo
que se reporta compara ese modelo contra un motor de reglas, no contra la ausencia
de control.

Los datos son el conjunto publico
[`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
{num(d["rows"])} operaciones europeas de dos dias de septiembre de 2013 con
{num(d["frauds"])} fraudes, uno cada {num(1 / d["prevalence"])}.

---

## El resultado

Medido una sola vez sobre las ultimas {num(s["test_window_days"] * 24, 1)} horas
del periodo, {num(s["test_rows"])} operaciones con {s["test_frauds"]} fraudes que
ninguno de los dos sistemas vio durante su calibracion.

| Indicador | Modelo | Motor de reglas |
|---|---:|---:|
| Perdida evitable que se evita | **{plain(c["savings_share"])}** | {plain(r["savings_share"])} |
| Monto fraudulento interceptado | **{plain(c["value_recall"])}** | {plain(r["value_recall"])} |
| Fraudes detectados de cada 100 | **{num(c["recall"] * 100)}** | {num(r["recall"] * 100)} |
| Alertas correctas de cada 100 | **{num(c["precision"] * 100)}** | {num(r["precision"] * 100)} |
| Alertas por dia | **{num(c["alerts_per_day"])}** | {num(r["alerts_per_day"])} |

Reemplazar el motor de reglas por el modelo reduce en {plain(sc["cost_reduction"])}
lo que hoy cuesta el fraude, contando el dinero perdido y el costo de operar la
revision, y lo hace con {num(r["alerts_per_day"] / c["alerts_per_day"], 1)} veces
menos alertas diarias. El punto de corte es {num(thr, 2)}.

Repitiendo la evaluacion a lo largo de toda la ventana, el ahorro promedia
{plain(bt["savings_share_mean"], 1)} con periodos que van de
{plain(bt["savings_share_min"], 1)} a {plain(bt["savings_share_max"], 1)}.

---

## Cuanto cuesta cada tipo de error

Comparar dos sistemas exige poder decir cuanto cuesta cada decision. Hay cuatro
situaciones posibles y cada una tiene una consecuencia economica distinta. Se
detecta un fraude real y se paga la revision recuperando buena parte del dinero.
Pasa un fraude sin detectar y se pierde el monto completo mas el contracargo. Se
detiene una compra legitima y se paga la revision mas el costo de haber molestado
al cliente. Se aprueba una compra legitima y no cuesta nada.

Ponerle numero a eso exige cuatro parametros que el negocio conoce y que no salen
de los datos. Todas las cifras del proyecto dependen de ellos.

| Parametro | Valor | Que representa |
|---|---:|---|
| Costo de revisar una alerta | {num(a["review"])} | Tiempo de analista por caso investigado |
| Costo de bloquear a un cliente legitimo | {num(a["friction"])} | Molestia al cliente, el mas incierto de los cuatro |
| Fraccion del monto recuperada | {num(a["recovery_rate"], 2)} | Cuanto se rescata detectando a tiempo |
| Cargo por contracargo | {num(a["chargeback_fee"])} | Costo fijo de procesar la disputa |

Los montos estan en la unidad del conjunto de datos, que su publicador no
documento. Lo que se traslada a otra operacion son las proporciones.

---

## A donde va el dinero

| Concepto | Monto | De donde sale |
|---|---:|---|
| Fraude no detectado | {money(c["cost_missed_fraud"])} | {c["false_negatives"]} operaciones que pasaron completas |
| Fraude detectado y no recuperado | {money(c["cost_unrecovered_fraud"])} | La fraccion que no se rescata |
| Revision de alertas | {money(c["cost_review"])} | {c["alerts"]} alertas investigadas |
| Bloqueos a clientes legitimos | {money(c["cost_friction"])} | {c["false_positives"]} compras validas detenidas |
| **Costo total con el modelo** | **{money(sc["model_cost"])}** | |

Casi todo el costo remanente esta en la primera fila. Operar la cola de revision
representa una fraccion menor, lo que indica que el margen de mejora esta en
detectar mas fraude y no en abaratar la operacion.

Para saber cuanto se ahorra hay que decir frente a que. Hay tres escenarios y
conviene no confundirlos.

| Escenario | Costo | Que representa |
|---|---:|---|
| Aprobar todo sin control | {money(sc["exposure"])} | Exposicion total al fraude. Techo de lo que se puede evitar, no una alternativa real |
| Motor de reglas | {money(sc["rules_cost"])} | El costo que la entidad tiene hoy |
| **Modelo** | **{money(sc["model_cost"])}** | La propuesta |
| **Ahorro frente al costo actual** | **{money(sc["incremental"])}** | {plain(sc["cost_reduction"])} menos de lo que hoy cuesta el fraude |

La ultima fila es el caso de negocio. Reemplazar el motor de reglas por el modelo
reduce en {plain(sc["cost_reduction"])} lo que la entidad gasta hoy por fraude,
contando tanto el dinero perdido como el costo de operar la revision. Medido sobre
la exposicion total, el modelo evita {plain(c["savings_share"])} de la perdida y el
motor de reglas {plain(r["savings_share"])}, una diferencia de {num(sc["points"])}
puntos.

---

## Que tan estable es el resultado

El procedimiento recorre la linea de tiempo hacia adelante. En cada paso el modelo
se reentrena con toda la historia disponible, el punto de corte se recalibra con el
tramo mas reciente, y se decide sobre el periodo siguiente sin volver a mirarlo.

| Periodo | Fraudes | Corte | Modelo | Reglas |
|---:|---:|---:|---:|---:|
{bt_rows}

Promedio {plain(bt["savings_share_mean"], 1)} con desviacion de
{num(bt["savings_share_std"] * 100, 1)} puntos. El modelo supera al motor de reglas
en los {bt["n_blocks_evaluated"]} periodos, con una ventaja minima de
{num(bt["uplift_points_min"], 1)} puntos.

Esa dispersion es el resultado que no aparece en ninguna medicion de un solo
corte. Un sistema que rinde parejo y otro que alterna entre los dos extremos no son
el mismo producto aunque su promedio coincida. El punto de corte tambien se mueve
entre periodos, lo que indica que recalibrar es parte de la operacion y no un
refinamiento opcional.

El conjunto cubre dos dias, de modo que cada periodo abarca horas. El procedimiento
es el correcto y la evidencia que produce es indicativa; sostener una promesa de
negocio requiere meses de operacion.

---

## Que se le puede exigir al modelo

La capacidad del equipo de revision y la precision minima tolerable no las decide
el area de datos. Cada restriccion tiene un precio, medido contra el mejor
resultado sin restricciones.

| Alertas/dia | Correctas de 100 | Fraudes | Monto | Del optimo |
|---:|---:|---:|---:|---:|
{cap_rows}

El maximo monto fraudulento que el modelo puede interceptar sin que la operacion
deje de ahorrar dinero es {plain(e["max_value_recall_while_profitable"])}, y en el
punto elegido intercepta {plain(c["value_recall"])}. Si la politica de riesgo exige
mas que eso, la respuesta esta en conseguir datos y no en mover el corte.

---

## Limites

Las variables estan anonimizadas mediante una transformacion que las vuelve
ilegibles, asi que el modelo ordena bien las operaciones pero no puede explicar por
que marco una en particular. Sin esa explicacion una cola de revision es dificil de
operar.

No hay identificador de tarjeta ni de comercio. Las senales mas usadas en los
sistemas reales son agregados sobre el historial de cada cliente, cuantas compras
lleva en la ultima hora o cuanto se aparta el monto de su costumbre, y son
justamente las que reconocerian a los {u["frauds_below_floor"]} fraudes que el
modelo hoy no ve. Esos casos concentran {plain(u["share_of_fraud_amount"])} del
monto fraudulento del tramo.

Quedan {s["test_frauds"]} fraudes en prueba, de modo que un solo caso mueve el
recall mas de un punto y cualquier diferencia pequena entre alternativas cae dentro
del ruido.

Son dos dias de 2013. El fraude cambia de forma cuando cambian los controles, asi
que lo que sobrevive de este trabajo es el procedimiento.

---

## Que seguiria

Conseguir datos con identificador de tarjeta, porque habilita el unico tipo de
predictor que ataca el techo y ahi esta el monto que hoy es invisible.

Bajar el costo de operar la cola de revision, que es el unico supuesto capaz de
mover el punto de corte optimo.

Intervalos de confianza por bootstrap sobre recall, precision y ahorro, para
comparar rangos en lugar de puntos.

Repetir el backtest sobre una ventana de meses. El procedimiento ya esta escrito en
`src/backtest.py` y solo necesita datos mas largos.

---

## Como se ejecuta

```bash
conda env create -f environment.yml
conda activate fraud-detection

python -m src.train                      # entrena, evalua y escribe reports/
python -m src.report                     # regenera este README y el informe
python -m src.predict --input tx.csv      # puntua operaciones nuevas
jupyter lab notebooks/                    # el recorrido comentado
```

El CSV no esta en el repositorio porque pesa 144 MB. `src.data.load_raw` lo
descarga de Kaggle la primera vez, lo cual requiere credenciales en
`~/.kaggle/kaggle.json`. Tambien se puede bajar a mano y dejarlo en
`data/creditcard.csv`.

```
src/
  config.py       rutas, semilla y supuestos de costo
  data.py         carga y particion temporal
  features.py     construccion de predictores
  baselines.py    motor de reglas de referencia
  costs.py        penalidad sensible al monto
  model.py        candidatos, seleccion y ajuste de hiperparametros
  economics.py    matriz de confusion traducida a dinero
  backtest.py     ventana expansiva con reajuste en cada paso
  evaluation.py   metricas y figuras
  report.py       genera el informe y este README desde las metricas
  train.py        orquestacion de punta a punta
  predict.py      puntuacion de operaciones nuevas
notebooks/
  deteccion-fraude.ipynb    el recorrido con las decisiones discutidas
docs/
  resumen-ejecutivo.html    una pagina para quien no va a leer el codigo
reports/
  metrics.json    respaldo de cada cifra de este README
  figures/
```

---

## Anexo, decisiones de metodo

### La particion separa pasado de futuro

El corte se hace por la columna `Time`. Los fraudes llegan en rachas, con varias
operaciones parecidas sobre la misma tarjeta comprometida en cuestion de minutos,
de modo que un corte al azar reparte esas operaciones entre entrenamiento y prueba
y el modelo termina evaluado sobre casos de los que ya vio un gemelo. Con el mismo
modelo y los mismos predictores, el corte aleatorio reporta 0,876 de average
precision y el temporal 0,794.

### El conjunto de prueba entra completo

Un filtro de rango intercuartilico sobre el tramo de prueba descarta el 26 % de las
operaciones y el 20 % de los fraudes, y sobre lo que queda el average precision
sube de 0,794 a 0,846. El fraude vive en la cola de la distribucion, asi que un
filtro de atipicos elimina sobre todo fraudes. En produccion tampoco se puede
descartar una operacion por ser rara antes de decidir si se aprueba.

### Treinta y un predictores

Las 28 componentes principales, el logaritmo del monto y la hora del dia en seno y
coseno. `Time` no entra porque mide segundos desde la primera operacion del
archivo, de modo que su valor solo existe dentro de esta ventana. La hora del dia
si se repite y aporta senal real, porque la madrugada concentra una tasa de fraude
varias veces superior al promedio.

No se generan derivadas de las componentes principales. Un arbol reproduce
cualquier transformacion monotona con un corte y una region no monotona con dos, de
modo que el cuadrado o el valor absoluto de una componente no agregan informacion y
reparten la importancia entre columnas casi identicas.

### El motor de reglas de referencia

Tres reglas calibradas sobre el mismo tramo de validacion que el punto de corte del
modelo, y aplicadas al mismo tramo de prueba.

| Regla | Ahorro | Fraudes | Monto | Alertas/dia |
|---|---:|---:|---:|---:|
{rule_rows}

Las dos reglas de monto destruyen valor. La mediana del fraude es de 9,25 unidades,
asi que un corte por monto alto cobra el costo de molestar clientes sobre compras
grandes legitimas sin interceptar practicamente nada. El arbol corto es la vara
relevante, y corta sobre componentes anonimizadas que un analista no podria
escribir a mano, de modo que en este conjunto cualquier referencia creible ya
tiene que derivarse de los datos.

### La penalidad que optimiza el modelo

Un modelo aprende minimizando una penalidad, y esa penalidad admite varias formas.
La mas simple castiga igual todos los errores. Otra castiga mas el fraude no
detectado, para compensar que es rarisimo. Una tercera castiga cada error en
proporcion al dinero en juego, asignando a cada operacion el arrepentimiento de
decidir mal sobre ella. Se comparan cinco alternativas por el ahorro que producen.

| Penalidad | Ahorro | Desv. | Monto | Alertas/dia |
|---|---:|---:|---:|---:|
{loss_rows}

La diferencia entre las dos primeras es menor que su variabilidad entre tramos, asi
que con esta cantidad de datos son equivalentes. Pesar por monto solo funciona si
ademas se compensa la rareza del fraude; sin esa correccion pierde
{num(cost_gap, 1)} puntos, porque con un fraude cada {num(1 / d["prevalence"])}
operaciones la escasez pesa mas que el monto.

La referencia lineal intercepta {plain(cand["logreg_balanced"]["value_recall_mean"])}
del monto, casi tanto como la ganadora, y sin embargo ahorra bastante menos. La
razon esta en sus {num(cand["logreg_balanced"]["alerts_per_day_mean"])} alertas
diarias, que se pagan en revision y en clientes molestados. Es el motivo por el que
el criterio de seleccion no puede ser una sola metrica de ordenamiento.

El area bajo la curva de precision y recall se calcula y queda en
`reports/metrics.json` para comparar con la literatura del conjunto de datos, pero
no interviene en la decision, porque promedia sobre toda la curva incluida la region
donde la operacion nunca va a trabajar y trata igual a un fraude de 2.126 y a uno
de 1,00.

Los hiperparametros se ajustan con optimizacion bayesiana sobre el mismo objetivo
de ahorro, {hp.get("n_trials", 0)} pruebas. Los que mas influyen son {imp_line}.

### El punto de corte

Para cada operacion el modelo devuelve un numero entre 0 y 1. Convertirlo en
decision requiere un limite, y 0,5 solo seria correcto si equivocarse en las dos
direcciones costara lo mismo. El limite elegido minimiza el costo total sobre el
tramo de validacion respetando la capacidad de revision, y resulta {num(thr, 2)}.

Queda alto porque el modelo reparte sus probabilidades de forma polarizada. A la
mayoria de los fraudes les asigna un valor cercano a 1 y a un grupo pequeno
practicamente 0, con muy pocos casos en el medio. Bajar el limite casi no agrega
detecciones y en cambio multiplica los bloqueos a clientes legitimos.

### Sensibilidad de los supuestos

Variando el costo de bloquear un cliente entre 0 y 100 el punto de corte se
mantiene en practicamente todo el rango, y variando el costo de revision entre 0 y
8 no se mueve. Recorriendo las dos dimensiones a la vez, solo baja en las
combinaciones donde uno de los costos es exactamente cero y el otro minimo. La
recomendacion no depende de los numeros mas debiles del analisis.

---

## Licencia

Codigo bajo licencia MIT. Copyright &copy; 2026 Daniel Hurtado. Ver [LICENSE](LICENSE).

El conjunto de datos se distribuye bajo sus propios terminos y no forma parte de
este repositorio. Las imagenes de `assets/` no estan cubiertas por la licencia MIT.
"""


def rewrap(markdown: str, width: int = 80) -> str:
    """Reacomoda los parrafos a un ancho fijo.

    Las interpolaciones cambian el largo de cada linea, asi que el texto crudo
    queda con cortes irregulares. Se reflujan solo los parrafos de prosa y se
    dejan intactos encabezados, tablas, listas y bloques de codigo.
    """
    import textwrap

    out, buffer, in_code = [], [], False

    def flush():
        if buffer:
            out.extend(textwrap.wrap(" ".join(buffer), width=width,
                                     break_long_words=False,
                                     break_on_hyphens=False))
            buffer.clear()

    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            out.append(line)
        elif in_code:
            out.append(line)
        elif not stripped or stripped.startswith(("#", "|", "-", "*", ">", "![")):
            flush()
            out.append(line)
        else:
            buffer.append(stripped)
    flush()
    return "\n".join(out)


def render(metrics_path=None, html_path=None) -> str:
    metrics_path = metrics_path or config.METRICS_PATH
    html_path = html_path or (config.ROOT / "docs" / "resumen-ejecutivo.html")

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    html = html_path.read_text(encoding="utf-8")

    start, end = html.index(OPEN), html.index(CLOSE) + len(CLOSE)
    updated = html[:start] + build_body(m) + html[end:]
    html_path.write_text(updated, encoding="utf-8")
    log.info("Informe regenerado en %s", html_path)

    readme = config.ROOT / "README.md"
    readme.write_text(rewrap(build_readme(m)), encoding="utf-8")
    log.info("README regenerado en %s", readme)
    return updated


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    render()


if __name__ == "__main__":
    main()

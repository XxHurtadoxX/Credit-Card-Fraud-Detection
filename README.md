# Deteccion de fraude con tarjeta de credito

Las entidades financieras pierden dinero por fraude con tarjeta, y la perdida no
se evita simplemente siendo mas estricto. Cada operacion exige decidir en el
momento si se aprueba o se detiene para revisarla, y las dos decisiones cuestan.
Detener una compra legitima molesta a un cliente que no hizo nada y consume
tiempo de un analista. Dejar pasar un fraude significa perder el monto completo
y pagar despues el contracargo.

La deteccion se apoya tradicionalmente en un motor de reglas escritas por
analistas de riesgo, del tipo detener la operacion si el monto pasa de cierto
valor o si combina varias caracteristicas sospechosas. Funciona para los
patrones evidentes y tiene la ventaja de que cualquiera puede explicar por que
se detuvo una compra, pero esta limitado por la cantidad de condiciones que una
persona puede escribir y mantener.

Este proyecto construye un modelo que estima la probabilidad de fraude de cada
operacion y decide alertar comparando lo que cuesta equivocarse en cada
direccion. El punto de corte no se elige por criterios estadisticos sino
economicos. Todo lo que se reporta compara ese modelo contra un motor de reglas,
no contra la ausencia de control.

Los datos son el conjunto publico
[`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
284.807 operaciones europeas de dos dias de septiembre de 2013 con 492 fraudes,
uno cada 579.

---

## El resultado

Medido una sola vez sobre las ultimas 7,7 horas del periodo, 56.962 operaciones
con 75 fraudes que ninguno de los dos sistemas vio durante su calibracion.

| Indicador | Modelo | Motor de reglas |
|---|---:|---:|
| Perdida evitable que se evita | **54 %** | 20 % |
| Monto fraudulento interceptado | **65 %** | 50 % |
| Fraudes detectados de cada 100 | **73** | 71 |
| Alertas correctas de cada 100 | **95** | 43 |
| Alertas por dia | **182** | 389 |

Reemplazar el motor de reglas por el modelo reduce en 43 % lo que hoy cuesta el
fraude, contando el dinero perdido y el costo de operar la revision, y lo hace
con 2,1 veces menos alertas diarias. El punto de corte es 0,96.

Repitiendo la evaluacion a lo largo de toda la ventana, el ahorro promedia
49,2 % con periodos que van de 16,9 % a 79,2 %.

---

## Cuanto cuesta cada tipo de error

Comparar dos sistemas exige poder decir cuanto cuesta cada decision. Hay cuatro
situaciones posibles y cada una tiene una consecuencia economica distinta. Se
detecta un fraude real y se paga la revision recuperando buena parte del dinero.
Pasa un fraude sin detectar y se pierde el monto completo mas el contracargo. Se
detiene una compra legitima y se paga la revision mas el costo de haber
molestado al cliente. Se aprueba una compra legitima y no cuesta nada.

Ponerle numero a eso exige cuatro parametros que el negocio conoce y que no
salen de los datos. Todas las cifras del proyecto dependen de ellos.

| Parametro | Valor | Que representa |
|---|---:|---|
| Costo de revisar una alerta | 4 | Tiempo de analista por caso investigado |
| Costo de bloquear a un cliente legitimo | 25 | Molestia al cliente, el mas incierto de los cuatro |
| Fraccion del monto recuperada | 0,85 | Cuanto se rescata detectando a tiempo |
| Cargo por contracargo | 15 | Costo fijo de procesar la disputa |

Los montos estan en la unidad del conjunto de datos, que su publicador no
documento. Lo que se traslada a otra operacion son las proporciones.

---

## A donde va el dinero

| Concepto | Monto | De donde sale |
|---|---:|---|
| Fraude no detectado | 2.979 | 20 operaciones que pasaron completas |
| Fraude detectado y no recuperado | 758 | La fraccion que no se rescata |
| Revision de alertas | 232 | 58 alertas investigadas |
| Bloqueos a clientes legitimos | 75 | 3 compras validas detenidas |
| **Costo total con el modelo** | **4.044** | |

Casi todo el costo remanente esta en la primera fila. Operar la cola de revision
representa una fraccion menor, lo que indica que el margen de mejora esta en
detectar mas fraude y no en abaratar la operacion.

Para saber cuanto se ahorra hay que decir frente a que. Hay tres escenarios y
conviene no confundirlos.

| Escenario | Costo | Que representa |
|---|---:|---|
| Aprobar todo sin control | 8.854 | Exposicion total al fraude. Techo de lo que se puede evitar, no una alternativa real |
| Motor de reglas | 7.065 | El costo que la entidad tiene hoy |
| **Modelo** | **4.044** | La propuesta |
| **Ahorro frente al costo actual** | **3.022** | 43 % menos de lo que hoy cuesta el fraude |

La ultima fila es el caso de negocio. Reemplazar el motor de reglas por el
modelo reduce en 43 % lo que la entidad gasta hoy por fraude, contando tanto el
dinero perdido como el costo de operar la revision. Medido sobre la exposicion
total, el modelo evita 54 % de la perdida y el motor de reglas 20 %, una
diferencia de 34 puntos.

---

## Que tan estable es el resultado

El procedimiento recorre la linea de tiempo hacia adelante. En cada paso el
modelo se reentrena con toda la historia disponible, el punto de corte se
recalibra con el tramo mas reciente, y se decide sobre el periodo siguiente sin
volver a mirarlo.

| Periodo | Fraudes | Corte | Modelo | Reglas |
|---:|---:|---:|---:|---:|
| 1 | 74 | 0,98 | 54,6 % | -15,9 % |
| 2 | 30 | 0,26 | 16,9 % | -232,6 % |
| 3 | 29 | 0,98 | 51,0 % | -29,3 % |
| 4 | 29 | 0,94 | 79,2 % | -37,8 % |
| 5 | 49 | 0,98 | 41,5 % | 24,5 % |
| 6 | 21 | 0,72 | 52,2 % | 25,0 % |

Promedio 49,2 % con desviacion de 18,4 puntos. El modelo supera al motor de
reglas en los 6 periodos, con una ventaja minima de 17,0 puntos.

Esa dispersion es el resultado que no aparece en ninguna medicion de un solo
corte. Un sistema que rinde parejo y otro que alterna entre los dos extremos no
son el mismo producto aunque su promedio coincida. El punto de corte tambien se
mueve entre periodos, lo que indica que recalibrar es parte de la operacion y no
un refinamiento opcional.

El conjunto cubre dos dias, de modo que cada periodo abarca horas. El
procedimiento es el correcto y la evidencia que produce es indicativa; sostener
una promesa de negocio requiere meses de operacion.

---

## Que se le puede exigir al modelo

La capacidad del equipo de revision y la precision minima tolerable no las
decide el area de datos. Cada restriccion tiene un precio, medido contra el
mejor resultado sin restricciones.

| Alertas/dia | Correctas de 100 | Fraudes | Monto | Del optimo |
|---:|---:|---:|---:|---:|
| 25 | 100 | 11 % | 6 % | 11 % |
| 50 | 100 | 21 % | 17 % | 27 % |
| 100 | 100 | 43 % | 42 % | 65 % |
| 150 | 98 | 63 % | 49 % | 77 % |
| 200 | 89 | 76 % | 66 % | 99 % |
| 400 | 48 | 81 % | 77 % | 79 % |
| 800 | 24 | 83 % | 77 % | 3 % |

El maximo monto fraudulento que el modelo puede interceptar sin que la operacion
deje de ahorrar dinero es 77 %, y en el punto elegido intercepta 65 %. Si la
politica de riesgo exige mas que eso, la respuesta esta en conseguir datos y no
en mover el corte.

---

## Limites

Las variables estan anonimizadas mediante una transformacion que las vuelve
ilegibles, asi que el modelo ordena bien las operaciones pero no puede explicar
por que marco una en particular. Sin esa explicacion una cola de revision es
dificil de operar.

No hay identificador de tarjeta ni de comercio. Las senales mas usadas en los
sistemas reales son agregados sobre el historial de cada cliente, cuantas
compras lleva en la ultima hora o cuanto se aparta el monto de su costumbre, y
son justamente las que reconocerian a los 13 fraudes que el modelo hoy no ve.
Esos casos concentran 23 % del monto fraudulento del tramo.

Quedan 75 fraudes en prueba, de modo que un solo caso mueve el recall mas de un
punto y cualquier diferencia pequena entre alternativas cae dentro del ruido.

Son dos dias de 2013. El fraude cambia de forma cuando cambian los controles,
asi que lo que sobrevive de este trabajo es el procedimiento.

---

## Que seguiria

Conseguir datos con identificador de tarjeta, porque habilita el unico tipo de
predictor que ataca el techo y ahi esta el monto que hoy es invisible.

Bajar el costo de operar la cola de revision, que es el unico supuesto capaz de
mover el punto de corte optimo.

Intervalos de confianza por bootstrap sobre recall, precision y ahorro, para
comparar rangos en lugar de puntos.

Repetir el backtest sobre una ventana de meses. El procedimiento ya esta escrito
en `src/backtest.py` y solo necesita datos mas largos.

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
operaciones parecidas sobre la misma tarjeta comprometida en cuestion de
minutos, de modo que un corte al azar reparte esas operaciones entre
entrenamiento y prueba y el modelo termina evaluado sobre casos de los que ya
vio un gemelo. Con el mismo modelo y los mismos predictores, el corte aleatorio
reporta 0,876 de average precision y el temporal 0,794.

### El conjunto de prueba entra completo

Un filtro de rango intercuartilico sobre el tramo de prueba descarta el 26 % de
las operaciones y el 20 % de los fraudes, y sobre lo que queda el average
precision sube de 0,794 a 0,846. El fraude vive en la cola de la distribucion,
asi que un filtro de atipicos elimina sobre todo fraudes. En produccion tampoco
se puede descartar una operacion por ser rara antes de decidir si se aprueba.

### Treinta y un predictores

Las 28 componentes principales, el logaritmo del monto y la hora del dia en seno
y coseno. `Time` no entra porque mide segundos desde la primera operacion del
archivo, de modo que su valor solo existe dentro de esta ventana. La hora del
dia si se repite y aporta senal real, porque la madrugada concentra una tasa de
fraude varias veces superior al promedio.

No se generan derivadas de las componentes principales. Un arbol reproduce
cualquier transformacion monotona con un corte y una region no monotona con dos,
de modo que el cuadrado o el valor absoluto de una componente no agregan
informacion y reparten la importancia entre columnas casi identicas.

### El motor de reglas de referencia

Tres reglas calibradas sobre el mismo tramo de validacion que el punto de corte
del modelo, y aplicadas al mismo tramo de prueba.

| Regla | Ahorro | Fraudes | Monto | Alertas/dia |
|---|---:|---:|---:|---:|
| revisar si Amount >= 2499.91 | -22 % | 0 % | 0 % | 213 |
| revisar si Amount >= 1814.92, o si Amount >= 71.30 entre las 0 y las 6 | -46 % | 0 % | 0 % | 442 |
| motor de reglas de profundidad 3, 8 hojas, umbral 0.98 | 20 % | 71 % | 50 % | 389 |

Las dos reglas de monto destruyen valor. La mediana del fraude es de 9,25
unidades, asi que un corte por monto alto cobra el costo de molestar clientes
sobre compras grandes legitimas sin interceptar practicamente nada. El arbol
corto es la vara relevante, y corta sobre componentes anonimizadas que un
analista no podria escribir a mano, de modo que en este conjunto cualquier
referencia creible ya tiene que derivarse de los datos.

### La penalidad que optimiza el modelo

Un modelo aprende minimizando una penalidad, y esa penalidad admite varias
formas. La mas simple castiga igual todos los errores. Otra castiga mas el
fraude no detectado, para compensar que es rarisimo. Una tercera castiga cada
error en proporcion al dinero en juego, asignando a cada operacion el
arrepentimiento de decidir mal sobre ella. Se comparan cinco alternativas por el
ahorro que producen.

| Penalidad | Ahorro | Desv. | Monto | Alertas/dia |
|---|---:|---:|---:|---:|
| Pesos por clase | 57,1 % | 7,8 | 72,4 % | 216 |
| Pesos por monto, reequilibrados | 55,6 % | 7,4 | 70,6 % | 214 |
| Sin ponderar | 54,9 % | 9,8 | 68,9 % | 180 |
| Pesos por monto | 45,3 % | 9,6 | 58,3 % | 195 |
| Referencia lineal | 38,2 % | 6,2 | 69,1 % | 456 |

La diferencia entre las dos primeras es menor que su variabilidad entre tramos,
asi que con esta cantidad de datos son equivalentes. Pesar por monto solo
funciona si ademas se compensa la rareza del fraude; sin esa correccion pierde
10,3 puntos, porque con un fraude cada 579 operaciones la escasez pesa mas que
el monto.

La referencia lineal intercepta 69 % del monto, casi tanto como la ganadora, y
sin embargo ahorra bastante menos. La razon esta en sus 456 alertas diarias, que
se pagan en revision y en clientes molestados. Es el motivo por el que el
criterio de seleccion no puede ser una sola metrica de ordenamiento.

El area bajo la curva de precision y recall se calcula y queda en
`reports/metrics.json` para comparar con la literatura del conjunto de datos,
pero no interviene en la decision, porque promedia sobre toda la curva incluida
la region donde la operacion nunca va a trabajar y trata igual a un fraude de
2.126 y a uno de 1,00.

Los hiperparametros se ajustan con optimizacion bayesiana sobre el mismo
objetivo de ahorro, 40 pruebas. Los que mas influyen son `learning_rate` 75 %,
`n_estimators` 21 %, `colsample_bytree` 2 %.

### El punto de corte

Para cada operacion el modelo devuelve un numero entre 0 y 1. Convertirlo en
decision requiere un limite, y 0,5 solo seria correcto si equivocarse en las dos
direcciones costara lo mismo. El limite elegido minimiza el costo total sobre el
tramo de validacion respetando la capacidad de revision, y resulta 0,96.

Queda alto porque el modelo reparte sus probabilidades de forma polarizada. A la
mayoria de los fraudes les asigna un valor cercano a 1 y a un grupo pequeno
practicamente 0, con muy pocos casos en el medio. Bajar el limite casi no agrega
detecciones y en cambio multiplica los bloqueos a clientes legitimos.

### Sensibilidad de los supuestos

Variando el costo de bloquear un cliente entre 0 y 100 el punto de corte se
mantiene en practicamente todo el rango, y variando el costo de revision entre 0
y 8 no se mueve. Recorriendo las dos dimensiones a la vez, solo baja en las
combinaciones donde uno de los costos es exactamente cero y el otro minimo. La
recomendacion no depende de los numeros mas debiles del analisis.

---

## Licencia

Codigo bajo licencia MIT. Copyright &copy; 2026 Daniel Hurtado. Ver
[LICENSE](LICENSE).

El conjunto de datos se distribuye bajo sus propios terminos y no forma parte de
este repositorio. Las imagenes de `assets/` no estan cubiertas por la licencia
MIT.

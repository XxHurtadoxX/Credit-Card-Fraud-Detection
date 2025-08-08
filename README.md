# Credit-Card-Fraud-Detection
Práctica de Ciencia de Datos

## Acerca del conjunto de datos

### Contexto
Es importante que las compañías de tarjetas de crédito reconozcan transacciones fraudulentas para que a los clientes no se les cobre por productos que no compraron.

### Contenido
El conjunto de datos contiene transacciones realizadas con tarjetas de crédito en septiembre de 2013 por tarjetahabientes europeos.

Incluye transacciones de solo dos días, con 492 fraudes de un total de 284,807 transacciones. El dataset está fuertemente desbalanceado: la clase positiva (fraudes) representa el 0.172% de todas las transacciones.

Solo contiene variables numéricas que son el resultado de una transformación PCA. Por motivos de confidencialidad no se pueden proporcionar las características originales ni más información contextual. Las variables V1, V2, …, V28 son los componentes principales obtenidos con PCA; las únicas variables que no han sido transformadas son `Time` y `Amount`.

La variable `Time` representa los segundos transcurridos entre cada transacción y la primera transacción del conjunto. La variable `Amount` es el importe de la transacción; puede usarse para enfoques de aprendizaje con coste dependiente del ejemplo. La variable `Class` es la variable objetivo y toma el valor 1 en caso de fraude y 0 en caso contrario.

### Métricas recomendadas
Dado el pronunciado desbalance de clases, se recomienda evaluar utilizando el Área Bajo la Curva Precisión-Recall (AUPRC). La exactitud (accuracy) basada únicamente en la matriz de confusión no es representativa en escenarios desbalanceados.

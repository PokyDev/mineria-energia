# PROYECTO INTEGRADOR — MINERÍA DE DATOS
## SEGUNDO AVANCE: Preparación de los datos y modelo supervisado
**Grupo 6 · Consumo energético**

Integrantes: Wendy Yolanny Jaimes Lugo · Jonathan David Quintero Villamizar · Javier Andrés Socha Maldonado
Asignatura: Minería de Datos · Programa: Ingeniería de Sistemas · Universitaria Tecnológica de Santander
Fecha: Noviembre de 2026

---

## 1. Introducción

En el primer avance se formuló la pregunta de investigación, se describió el conjunto Appliances Energy Prediction
(UCI) y se hizo su análisis exploratorio. Este segundo avance corresponde a la semana 13 del cronograma, "Limpieza,
transformación y construcción del primer modelo", y desarrolla las fases de **Preparación de los datos** y
**Modelado** de la metodología CRISP-DM.

El trabajo se dividió en dos cuadernos de Jupyter. `02_preparacion.ipynb` limpia, transforma y divide los datos.
`03_modelo_supervisado.ipynb` entrena y evalúa los modelos de regresión. Todas las cifras de este documento salen de la
ejecución de esos cuadernos. Como pidió la guía para el Grupo 6, todo el proceso respeta el **orden temporal** de los
registros.

Recordamos la pregunta de investigación: *¿En qué medida las variables ambientales interiores y exteriores, junto con
la información temporal disponible, permiten estimar el consumo energético de los electrodomésticos (Appliances) y
reconocer grupos de patrones de uso?* Este avance responde a la primera parte de la pregunta, la estimación del
consumo.

## 2. Objetivos del avance

- Verificar la calidad de los datos y documentar cada decisión de limpieza.
- Justificar las variables que se excluyen del modelo.
- Construir atributos temporales y aplicar la codificación y el escalado que correspondan, ajustando las
  transformaciones solo con el conjunto de entrenamiento.
- Dividir los datos en entrenamiento y prueba con una estrategia cronológica.
- Entrenar y evaluar un modelo de regresión con métricas adecuadas, justificar sus parámetros y presentar sus
  predicciones.

## 3. Preparación de los datos

### 3.1 Carga y verificación de calidad

Los datos se descargaron directamente de UCI con la librería `ucimlrepo` (id 374). La columna `date` llega con espacios
intercalados, así que se limpió antes de convertirla a fecha, y los registros se ordenaron cronológicamente. Se guardó
una copia del conjunto original (`data/appliances_original.csv`), lo que permite la licencia CC BY 4.0 citando a los
autores.

La verificación automática confirmó lo encontrado en la semana 12: los 19 735 registros no tienen valores nulos, filas
duplicadas ni marcas de tiempo repetidas, y todos los intervalos entre registros son de exactamente 10 minutos. Se
comprobó además que las variables están dentro de rangos físicos posibles: humedades entre 0 y 100 %, temperaturas
entre −6,6 y 29,9 °C y consumo siempre positivo. Como no hay datos faltantes ni duplicados, **no fue necesario imputar
ni eliminar registros**.

Para que ninguna decisión de preparación dependiera del periodo de prueba, la evidencia de las secciones 3.2 y 3.3 se
calculó solo con el tramo de entrenamiento.

### 3.2 Variables excluidas

**Variables aleatorias `rv1` y `rv2`.** Los autores las añadieron a propósito para probar métodos de selección de
variables. Se comprobó que son idénticas en todas las filas y que su correlación con el consumo en el entrenamiento es
prácticamente nula (r = −0,007). No contienen información de la vivienda, por lo que se excluyeron.

**Consumo de iluminación `lights`.** En el entrenamiento, el 74,3 % de sus registros vale cero y solo toma ocho valores
distintos (0 a 70 Wh). Aunque es la variable con mayor correlación lineal con el objetivo (r = 0,218), se excluyó por
tres razones. Primero, es una medición de consumo eléctrico tomada en el mismo instante que `Appliances`: en un
escenario real no estaría disponible antes de conocer el consumo del hogar, así que usarla sería **fuga de
información**. Segundo, la pregunta de investigación busca estimar el consumo a partir de condiciones ambientales y del
tiempo, no de otros consumos. Tercero, su relación con el objetivo refleja la ocupación de la vivienda (cuando hay luces
encendidas, el consumo medio sube de 84,3 a 137,7 Wh), y esa ocupación ya queda representada de forma legítima por la
hora y el día.

**Humedad exterior norte `RH_6`.** El 9,9 % de los registros (1 949) marca una humedad menor a 5 %, con un mínimo de
1 %. En esos mismos instantes, la estación meteorológica (`RH_out`) registró humedades de entre 24 % y 83,7 %, con una
media de 52,3 %. Una humedad exterior cercana a 1 % no es creíble en ese clima, así que se interpretó como una **falla
del sensor**. En el entrenamiento estas lecturas no aparecen entre enero y marzo y suben al 10,8 % en abril: el sensor
se estaba degradando. Se consideraron dos alternativas. Eliminar las filas descartaría registros cuyo consumo y demás
sensores son válidos, y además rompería la continuidad de la serie. Imputar con la mediana de entrenamiento
introduciría un valor invernal constante en una fracción cada vez mayor de los registros de primavera. Por eso se
decidió **excluir la variable**: la humedad exterior ya la cubre `RH_out` y la temperatura del mismo lado la cubre `T6`,
que funciona correctamente.

**Fecha cruda `date`.** No se usa como predictor, porque un modelo no puede extrapolar fechas futuras. Se reemplazó por
atributos temporales (sección 3.4) y se conservó en los archivos solo para ordenar y graficar.

### 3.3 Valores atípicos y transformación del objetivo

Con la regla del rango intercuartílico calculada sobre el entrenamiento, el límite superior es 175 Wh y 1 776 registros
(11,2 %) lo superan. Esos valores se concentran en las horas de actividad del hogar (mañana y tarde-noche) y casi no
aparecen de madrugada. Son **picos reales de uso** de electrodomésticos, no errores de medición, y son justamente lo que
interesa estimar, así que no se eliminaron ni se recortaron.

Para reducir la influencia de esos picos y la fuerte asimetría a la derecha de `Appliances` (coeficiente de asimetría
de 3,32), los modelos se entrenaron sobre `log1p(Appliances)`, que baja la asimetría a 1,15. Las predicciones se
devuelven a Wh con la función inversa `expm1`. Esta transformación no aprende parámetros de los datos, así que no
genera fuga de información, y se aplicó dentro del modelo con `TransformedTargetRegressor` de scikit-learn.

*[Figura sugerida: histogramas de Appliances y log1p(Appliances) y horas de los atípicos — notebook 02, sección 5.]*

### 3.4 Atributos temporales y codificación

El análisis exploratorio mostró que el consumo cambia mucho a lo largo del día: la media más alta está a las 18:00 y la
más baja a las 03:00. A partir de `date` se derivaron dos tipos de atributos:

- **Hora cíclica (`hora_sin`, `hora_cos`).** La hora del día, con sus minutos, se codificó con seno y coseno para que
  las 23:50 y las 00:00 queden cerca. Con la hora como número simple, el modelo las vería separadas por 23 unidades.
- **Día de la semana (`dia_semana`).** Es una variable categórica, porque no hay un orden de magnitud entre los días,
  así que se codificó con *one-hot* (siete columnas binarias). Esa codificación ya incluye la información de fin de
  semana que se planteó en el primer avance, por lo que no se añadió una variable de fin de semana redundante.

Cada atributo se calcula con la marca de tiempo de su propia fila, sin usar registros futuros, así que no introduce
fuga de información.

### 3.5 División cronológica entrenamiento / prueba

Se mantuvo la división cronológica 80/20 definida en el primer avance. El **entrenamiento** va del 11 de enero al 30 de
abril de 2016 (15 788 registros) y la **prueba** del 30 de abril al 27 de mayo de 2016 (3 947 registros). No se usó una
división aleatoria. Registros separados por 10 minutos son casi idénticos, y mezclarlos pondría en el entrenamiento
datos "vecinos" de los de prueba, lo que daría una evaluación demasiado optimista. La división cronológica simula el uso
real: aprender con el pasado y predecir el futuro.

*[Figura sugerida: consumo medio diario con la frontera entrenamiento / prueba — notebook 02, sección 7.]*

### 3.6 Escalado y prevención de fuga de información

Las variables numéricas tienen escalas muy distintas (°C, %, mm Hg, km/h), y la regresión regularizada necesita que
todas estén en la misma escala para penalizarlas por igual. Por eso se definió un preprocesador (`ColumnTransformer`)
que aplica `StandardScaler` a las 25 variables numéricas y `OneHotEncoder` a `dia_semana`. El resultado son 32 columnas.

El preprocesador se guardó **sin ajustar** y se colocó dentro de un `Pipeline` junto a cada modelo. Así, las medias y
desviaciones del escalado se aprenden únicamente con los datos de entrenamiento de cada partición de validación y, al
final, con el entrenamiento completo. Se verificó que las medias aprendidas coinciden exactamente con las del
entrenamiento y no con las de prueba.

Esa comparación dejó ver un hecho que condiciona el modelado: **el periodo de prueba es más cálido que el de
entrenamiento**. Por ejemplo, `T6` promedia 6,1 °C en entrenamiento y 15,0 °C en prueba, y `T1` pasa de 21,2 a 23,8 °C.

### 3.7 Resumen de la preparación

| Variable / paso | Decisión | Justificación |
|---|---|---|
| `date` | Reemplazada por `hora_sin`, `hora_cos` y `dia_semana` | No extrapolable; lo útil es la hora y el día |
| `rv1`, `rv2` | Excluidas | Variables aleatorias de los autores, idénticas y sin relación con el objetivo |
| `lights` | Excluida | Consumo simultáneo al objetivo (fuga) y casi constante |
| `RH_6` | Excluida (no se imputa) | Sensor con lecturas implausibles y fallas crecientes; cubierta por `RH_out` y `T6` |
| Nulos y duplicados | No hay; no se imputa ni se elimina | Verificado automáticamente |
| Atípicos de `Appliances` | Se conservan | Picos reales de uso en horas de actividad |
| `Appliances` | `log1p` al entrenar, `expm1` al predecir | Asimetría fuerte a la derecha |
| `dia_semana` | One-hot | Categórica sin orden |
| Numéricas | `StandardScaler` ajustado solo con entrenamiento | Escalas distintas; requerido por la regresión regularizada |
| División | Cronológica 80/20 | Serie temporal: entrenar con el pasado y probar con el futuro |

El conjunto preparado queda con **26 predictores**: las temperaturas `T1`–`T9`, las humedades `RH_1`–`RH_9` excepto
`RH_6`, seis variables meteorológicas, la hora cíclica y el día de la semana. Se exportó como
`data/train_preparado.csv` y `data/test_preparado.csv`.

## 4. Modelo supervisado: regresión del consumo

### 4.1 Variable objetivo y enfoque

La variable objetivo es `Appliances`, el consumo de los electrodomésticos en Wh cada 10 minutos. Es una variable
continua, así que la tarea es de **regresión**. Se compararon tres modelos: una línea base ingenua, una regresión
lineal regularizada (Ridge) y un Random Forest.

### 4.2 Estrategia de validación y métricas

Los parámetros se eligieron con validación cruzada temporal (`TimeSeriesSplit`, 5 particiones) **dentro** del
entrenamiento. Cada partición entrena con un bloque inicial y valida con el bloque siguiente: la primera entrena del 11
al 29 de enero y valida hasta el 17 de febrero, y la última entrena hasta el 12 de abril y valida hasta el 30 de abril.
El conjunto de prueba no se usó para ninguna decisión y se evaluó una sola vez al final.

Todas las métricas se calculan en Wh. Se usaron tres:

- **MAE** (error absoluto medio): cuántos Wh se equivoca el modelo en promedio. Fue el criterio para elegir los
  parámetros, por ser la medida más directa del error y la menos dominada por unos pocos picos.
- **RMSE** (raíz del error cuadrático medio): penaliza más los errores grandes, es decir, los picos mal estimados.
- **R²**: la proporción de la variabilidad del consumo que explica el modelo.

### 4.3 Línea base

Como referencia mínima, la línea base predice para cada registro de prueba el consumo medio de esa hora del día en el
entrenamiento. En la prueba obtuvo MAE = 46,9 Wh, RMSE = 85,5 Wh y R² = 0,118. Un modelo solo aporta valor si mejora
esta referencia.

### 4.4 Modelo 1: regresión Ridge

Ridge es una regresión lineal con penalización L2. Es el modelo lineal e interpretable que se propuso en el primer
avance. Se prefirió a los mínimos cuadrados simples porque muchas temperaturas y humedades están fuertemente
correlacionadas entre sí, y la penalización estabiliza los coeficientes. El parámetro `alpha`, que regula la fuerza de
la penalización, se buscó en una rejilla logarítmica de 10⁻³ a 10⁵. El mejor valor fue **alpha = 1000**, con un MAE de
validación de 45,1 Wh. Que se necesite una penalización alta indica que hay mucha redundancia entre los predictores
ambientales.

*[Figura sugerida: MAE según alpha en entrenamiento y validación — notebook 03, sección 3.]*

### 4.5 Modelo 2: Random Forest

Para comprobar si un modelo no lineal captura mejor la relación entre ambiente, hora y consumo, se entrenó un Random
Forest. Se fijaron 200 árboles, suficientes para un promedio estable, y `max_features = 0,5`, que reduce la correlación
entre árboles, con `random_state = 42`. Se buscaron `max_depth` y `min_samples_leaf`, los dos parámetros que controlan
el sobreajuste:

| max_depth | min_samples_leaf | MAE entrenamiento (Wh) | MAE validación (Wh) |
|---|---|---|---|
| **16** | **50** | **40,7** | **45,6** |
| 4 | 10 | 44,8 | 45,6 |
| 8 | 50 | 41,7 | 45,6 |
| 4 | 50 | 45,6 | 45,7 |
| 8 | 10 | 36,9 | 46,3 |
| 16 | 10 | 30,8 | 46,9 |

Las configuraciones con hojas pequeñas y árboles profundos tienen un error de entrenamiento mucho menor que el de
validación: memorizan el ruido de cada intervalo de 10 minutos. Se eligió `max_depth = 16` y `min_samples_leaf = 50`,
aunque las cuatro mejores configuraciones quedan prácticamente empatadas, en torno a 45,6 Wh.

### 4.6 Resultados en el conjunto de prueba

| Modelo | MAE (Wh) | RMSE (Wh) | R² | MAE entrenamiento | MAE validación cruzada | Mejora del MAE frente a la línea base |
|---|---|---|---|---|---|---|
| Línea base (media por hora) | 46,9 | 85,5 | 0,118 | — | — | — |
| **Ridge (alpha = 1000)** | **40,9** | 86,1 | 0,105 | 45,0 | 45,1 | **13,0 %** |
| Random Forest | 51,5 | 86,7 | 0,094 | 37,2 | 45,6 | −9,8 % |

Se eligió **Ridge**, porque tiene el menor MAE de validación cruzada. La elección no se hizo con la prueba, para no
usar ese periodo en las decisiones. En la prueba, Ridge reduce el error absoluto medio en un 13 % frente a la línea
base: se equivoca en promedio 40,9 Wh, para un consumo medio real de 96,4 Wh. Su error de entrenamiento (45,0 Wh) y su
error de validación (45,1 Wh) casi coinciden, lo que indica que **no hay sobreajuste**.

Ridge, en cambio, no mejora el RMSE ni el R² de la línea base (86,1 frente a 85,5 Wh, y 0,105 frente a 0,118). La
sección 4.8 explica esta diferencia entre métricas.

El Random Forest queda por debajo de la línea base en la prueba. Su MAE pasa de 37,2 Wh en entrenamiento y 45,6 Wh en
validación a 51,5 Wh en prueba. La causa principal es el cambio de estación descrito en la sección 3.6: el 16 % de los
registros de prueba tiene una temperatura `T6` mayor que el máximo observado en entrenamiento, y los árboles de decisión
no pueden extrapolar fuera de los rangos que conocen.

### 4.7 Predicciones

La tabla muestra una muestra de predicciones sobre el periodo de prueba (un registro cada 400). El archivo completo
está en `data/predicciones_test.csv`.

| Fecha y hora | Real (Wh) | Línea base | Ridge | Random Forest |
|---|---|---|---|---|
| 30-04-2016 08:20 | 370 | 102,0 | 63,5 | 68,2 |
| 03-05-2016 03:00 | 50 | 47,2 | 49,5 | 55,0 |
| 05-05-2016 21:40 | 50 | 101,4 | 62,8 | 80,4 |
| 08-05-2016 16:20 | 60 | 116,4 | 116,8 | 132,2 |
| 11-05-2016 11:00 | 50 | 135,2 | 121,0 | 169,9 |
| 14-05-2016 05:40 | 60 | 50,7 | 43,3 | 68,2 |
| 17-05-2016 00:20 | 50 | 52,7 | 56,1 | 61,9 |
| 19-05-2016 19:00 | 70 | 149,7 | 80,6 | 152,6 |
| 22-05-2016 13:40 | 100 | 128,3 | 143,1 | 146,1 |
| 25-05-2016 08:20 | 60 | 102,0 | 77,1 | 99,7 |

Los modelos siguen bien el patrón diario: bajo consumo de madrugada y más alto en las horas de actividad. Sin embargo,
no anticipan los picos puntuales. En el primer registro, por ejemplo, el consumo real fue de 370 Wh y Ridge predijo
63,5 Wh. Las predicciones de Ridge nunca superan los 172 Wh, mientras que en la prueba hay consumos de cientos de Wh.

*[Figuras sugeridas: consumo real vs. predicho en la primera semana de prueba (notebook 03, sección 6.2) y dispersión
real vs. predicho (sección 6.3).]*

### 4.8 Análisis de errores

| Nivel de consumo real | Registros | Error medio de Ridge (Wh) | MAE de Ridge (Wh) | MAE de la línea base (Wh) | MAE del Random Forest (Wh) |
|---|---|---|---|---|---|
| ≤ 100 Wh | 3 120 | −16,4 | 20,4 | 30,1 | 38,0 |
| 100–200 Wh | 502 | +27,5 | 33,6 | 34,6 | 36,4 |
| > 200 Wh | 325 | +248,5 | 248,5 | 227,3 | 205,0 |

El error medio se calcula como el valor real menos el predicho, así que un valor positivo indica que el modelo
**subestima**. Ridge es claramente mejor que la línea base en los registros de consumo bajo, que son el 79 % de la
prueba. En los picos de más de 200 Wh, en cambio, subestima de forma sistemática, en unos 248 Wh. Esto pasa porque, al
entrenar sobre el logaritmo, el modelo tiende a predecir valores cercanos al consumo típico (la mediana es 60 Wh).

Esos 325 registros (8 % de la prueba) concentran cerca del 89 % del error cuadrático de Ridge. Por eso Ridge gana en
MAE, que pesa por igual cada registro, pero no en RMSE ni en R², que están dominados por los picos. Por horas, el mayor
error se da a las 17:00 (MAE de 109 Wh en Ridge) y el menor de madrugada (unos 10 Wh a las 04:00).

*[Figura sugerida: MAE y error medio por hora del día — notebook 03, sección 6.4.]*

### 4.9 Importancia de las variables

La importancia por permutación mide cuánto empeora el MAE en la prueba al desordenar al azar cada variable. En ambos
modelos, la **hora del día** es por mucho la variable más importante: al permutar `hora_sin` o `hora_cos`, el MAE
empeora entre 5,0 y 5,9 Wh. Le siguen, en Ridge, algunas variables interiores: `RH_1` (4,2 Wh), `T9` (3,9 Wh), `T3`
(2,5 Wh) y `RH_9` (1,7 Wh). El clima exterior y el día de la semana aportan muy poco (menos de 0,5 Wh).

Los coeficientes de Ridge están estandarizados y en escala logarítmica. Asocian un mayor consumo a temperaturas más
altas en `T3` (lavandería) y `T8`, y a más humedad en `RH_1` (cocina) y `RH_3`. Tiene sentido: cocinar o lavar eleva la
temperatura y la humedad locales. Por la colinealidad entre sensores, los coeficientes individuales deben leerse con
cautela y no como relaciones causales.

*[Figuras sugeridas: coeficientes de Ridge (notebook 03, sección 7.1) e importancia por permutación (sección 7.2).]*

## 5. Discusión preliminar

Los resultados responden de forma parcial a la primera parte de la pregunta de investigación. Con variables ambientales
y de calendario, y evaluando sobre un periodo futuro, el mejor modelo explica alrededor del 10 % de la variabilidad del
consumo y reduce el error absoluto medio en un 13 % frente a una regla simple basada en la hora. La mayor parte de la
información útil está en la **hora del día**. Los sensores ambientales aportan una mejora pequeña, sobre todo los de la
cocina y la lavandería, que reflejan indirectamente la actividad de los ocupantes.

El consumo de los electrodomésticos depende principalmente de **acciones humanas puntuales**, como encender el horno o
la lavadora, que estos sensores no registran directamente. Por eso los modelos estiman bien el consumo de base, pero no
anticipan los picos. El bajo R² no se debe a un error del procedimiento: es el resultado de una evaluación exigente,
que separa el pasado del futuro y no deja que el modelo vea datos de prueba. Además, ese futuro trae un cambio de
estación (de invierno a primavera) que el entrenamiento casi no cubre.

## 6. Limitaciones

- Los datos corresponden a una sola vivienda en Bélgica durante 4,5 meses. Los resultados no se pueden generalizar a
  otros hogares ni a otros climas.
- El periodo de prueba (mayo) es más cálido que el de entrenamiento. Parte del error se debe a ese cambio de
  distribución y no solo al modelo.
- No se usaron rezagos del propio consumo, por ejemplo el consumo de los 10 o 60 minutos anteriores. Mejorarían la
  predicción, pero convertirían el problema en un pronóstico autorregresivo, distinto de la pregunta planteada.
- Las correlaciones y los coeficientes describen asociaciones, no causalidad.
- Al excluir `lights` y `RH_6` se pierde información potencialmente útil. Se priorizó evitar la fuga de información y
  los datos de un sensor defectuoso.

## 7. Plan para el siguiente avance (semana 14)

| Etapa | Actividad pendiente | Evidencia esperada |
|---|---|---|
| Clustering | Seleccionar variables de patrón de uso, escalar y aplicar K-Means con distintos valores de k | Perfiles de los grupos y coeficiente de silueta |
| Evaluación | Profundizar el análisis de errores del modelo supervisado | Interpretación de métricas, sesgos y limitaciones |
| Interpretación | Comparar la utilidad de ambos enfoques sin equiparar sus métricas | Conclusiones sustentadas en los resultados |
| Video | Preparar el guion y grabar, con participación de todos los integrantes | Guion y grabación |

## 8. Conclusiones del segundo avance

Los datos quedaron preparados sin perder registros. Se excluyeron cuatro variables con justificación (`rv1`, `rv2`,
`lights` y `RH_6`), se conservaron los picos reales de consumo, se transformó el objetivo con `log1p` y se construyeron
atributos de hora cíclica y día de la semana. La división es cronológica, y todas las transformaciones que aprenden
parámetros se ajustaron solo con el entrenamiento, dentro de un `Pipeline`, lo que evita la fuga de información.

Como primer modelo se seleccionó una **regresión Ridge (alpha = 1000)**, elegida por validación cruzada temporal. En el
periodo de prueba obtuvo MAE = 40,9 Wh, RMSE = 86,1 Wh y R² = 0,105. Mejora en un 13 % el error absoluto medio de la
línea base, pero no la mejora en RMSE ni en R², porque subestima los picos de consumo. El Random Forest no generalizó
al periodo más cálido de la prueba. Estos resultados muestran que la hora del día es el principal factor explicativo y
que las condiciones ambientales, por sí solas, permiten una estimación limitada del consumo. Esa es una conclusión
relevante para la pregunta de investigación.

## 9. Reproducibilidad y uso de IA

**Archivos entregados** (carpeta `docs/Semana_13/`):

- `02_preparacion.ipynb`: preparación de los datos.
- `03_modelo_supervisado.ipynb`: modelos de regresión y su evaluación.
- `data/appliances_original.csv`: conjunto original.
- `data/train_preparado.csv` y `data/test_preparado.csv`: conjuntos preparados.
- `data/predicciones_test.csv`: predicciones del periodo de prueba.
- `preprocesador.joblib`: definición del preprocesador.

**Instrucciones de ejecución:** con Python 3.13 y `uv`, ejecutar `uv sync` y luego `uv run jupyter lab`. Correr primero
`02_preparacion.ipynb` (requiere internet para descargar los datos de UCI) y después `03_modelo_supervisado.ipynb`.
Todo proceso aleatorio usa `random_state = 42`, así que los resultados son reproducibles. Librerías: pandas, numpy,
matplotlib, seaborn, scikit-learn y ucimlrepo.

**Uso de IA:** se usó un asistente de IA (Claude, de Anthropic) como apoyo para estructurar el código de los cuadernos y
redactar el borrador de este documento. El equipo ejecutó personalmente los cuadernos y verificó que todas las cifras
reportadas corresponden a las salidas obtenidas. Las decisiones metodológicas fueron revisadas y discutidas por los
integrantes.

## 10. Referencias

Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5VC8G. Licencia CC BY 4.0.

Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a
low-energy house. *Energy and Buildings, 140*, 81–97.

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*,
2825–2830.

Universitaria Tecnológica de Santander (UTS). (s. f.). *Proyecto Integrador: Minería de Datos*. Guía del proyecto
integrador, tercer corte (documento suministrado por el equipo).

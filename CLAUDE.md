# CLAUDE.md — Proyecto integrador de Minería de Datos

## Proyecto
Minería de datos sobre el dataset **Appliances Energy Prediction** (UCI id **374**).
Objetivo: predecir y entender el consumo energético de los electrodomésticos de una vivienda
a partir de condiciones internas (temperatura/humedad por habitación) y meteorológicas externas.

- Dataset: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- Repo original (código R, paper, info adicional): https://github.com/LuisM78/Appliances-energy-prediction-data
- Paper de referencia: Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in a low-energy house*. Energy and Buildings, 140, 81–97.

## Datos (verificar en el notebook, no asumir)
- **Licencia:** CC BY 4.0 (citar a los autores).
- **Origen:** casa de bajo consumo en Stambruges, Bélgica. Sensores ZigBee (T/RH interiores),
  medidores m-bus (energía) y clima de la estación del aeropuerto de Chièvres.
- **Tamaño:** 19 735 filas × 29 columnas, registros cada **10 min**, ~4,5 meses (ene–may 2016). Sin nulos reportados.
- **Objetivo:** `Appliances` (Wh, continua, muy sesgada a la derecha → revisar log-transform).
- **Variables:**
  - `date` — marca temporal (ucimlrepo la entrega con formato raro; se parsea quitando espacios, ver notebook 01).
  - `lights` — Wh de luces (mayoría ceros; candidata a excluir o binarizar).
  - `T1..T9`, `RH_1..RH_9` — temperatura (°C) y humedad (%) por zona (T6/RH_6 = exterior norte, T9 = parents room…).
  - `T_out`, `Press_mm_hg`, `RH_out`, `Windspeed`, `Visibility`, `Tdewpoint` — clima externo.
  - `rv1`, `rv2` — **variables aleatorias** añadidas por los autores para probar selección de variables → excluir del modelo y justificarlo.

## Stack y ejecución
- Python 3.13, gestionado con **uv** (`pyproject.toml`, `uv.lock`). `python` no está en PATH: usar `uv run ...`.
- Librerías: pandas, numpy, matplotlib, seaborn, scikit-learn, ucimlrepo, jupyterlab.
- Abrir notebooks: `uv sync` y luego `uv run jupyter lab`.
- Carga de datos: `fetch_ucirepo(id=374)` (requiere internet).
- Notebooks numerados por etapa: `01_comprension_exploración.ipynb`, `02_...`, etc.

## Convenciones de trabajo
- Todo en **español** (markdown, comentarios, gráficos).
- Datos son **serie temporal**: ordenar por `date`; split train/test **cronológico** (no aleatorio) o `TimeSeriesSplit`.
- **Evitar fuga de información:** escaladores, imputadores y transformaciones se ajustan (`fit`) solo con train,
  idealmente dentro de un `Pipeline` de scikit-learn.
- Fijar `random_state` en todo lo estocástico.
- Cada decisión (variable excluida, parámetro, transformación) se justifica en una celda markdown.
- Uso de IA: se documenta en el informe y el estudiante verifica personalmente los resultados.

## Checklist de requisitos del proyecto

### 1. Comprensión del problema
- [x] Formular una pregunta investigable
- [x] Describir origen del dataset
- [x] Licencia / condiciones de uso
- [x] Número de filas y variables
- [x] Variable objetivo
- [x] Pertinencia del dataset para la pregunta

### 2. Exploración
- [x] Tipos de variables
- [x] Estadísticas descriptivas
- [x] Visualizaciones
- [x] Distribución de la variable objetivo
- [x] Análisis de nulos
- [x] Análisis de duplicados
- [x] Análisis de valores atípicos

### 3. Preparación
- [x] Documentar limpieza
- [x] Imputación (si corresponde)
- [x] Codificación (si corresponde)
- [x] Escalado (si corresponde)
- [x] Justificar variables excluidas
- [x] Transformaciones ajustadas solo con entrenamiento (sin fuga de información)

### 4. Modelo supervisado
- [x] Definir variable objetivo
- [x] División entrenamiento/prueba con estrategia apropiada
- [x] Entrenar regresión o clasificación
- [x] Evaluar con métricas adecuadas
- [x] Justificar parámetros
- [x] Presentar predicciones

### 5. Modelo no supervisado
- [ ] Seleccionar variables pertinentes
- [ ] Escalar cuando corresponda
- [ ] Aplicar KMeans u otro clustering justificado
- [ ] Mostrar los grupos
- [ ] Evaluar coherencia (p. ej. silueta)

### 6. Análisis e interpretación
- [ ] Interpretar métricas
- [ ] Explicar errores, sesgos y limitaciones
- [ ] Comparar utilidad de ambos enfoques (sin equiparar métricas de tareas distintas)

### 7. Reproducibilidad
- [ ] Adjuntar datos
- [ ] Adjuntar código / notebooks
- [ ] Instrucciones de ejecución
- [ ] Referencias
- [ ] Declarar uso de IA y verificación personal de resultados

## Checklist de entregables (marcar al entregar)
- [x] **Semana 12** — Aprobación del problema, reconocimiento y análisis exploratorio del dataset.
- [ ] **Semana 13** — Limpieza, transformación y construcción del primer modelo.
- [ ] **Semana 14** — Segundo modelo, evaluación, interpretación, guion y grabación.
- [ ] **Semana 15** — Entrega simultánea del informe, archivos reproducibles y video.
- [ ] **Semana 16** — Sustentación individual y retroalimentación final.

## Estado actual
- Semana 12 completa: `docs/Semana_12/` (notebook del grupo + PDF del primer avance).
- Semana 13 lista (pendiente de entregar): `docs/Semana_13/02_preparacion.ipynb`, `03_modelo_supervisado.ipynb`,
  `informe_semana13.md`, `data/` y `preprocesador.joblib`.
  Decisiones: excluir rv1/rv2, lights (fuga), RH_6 (sensor falla desde abril, 38 % del test) y date cruda;
  log1p del objetivo; hora cíclica + one-hot de dia_semana; StandardScaler en Pipeline; split cronológico 80/20.
  Resultado: Ridge (alpha=1000) MAE 40,9 Wh (−13 % vs. base por hora), R² 0,105; Random Forest peor que la base
  (test más cálido, no extrapola).
- Próximo paso: semana 14 (sección 5 clustering, sección 6 interpretación, guion/video).

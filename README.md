# Minería de datos: consumo energético de electrodomésticos

Proyecto integrador de **Minería de Datos** (Grupo 6, Ingeniería de Sistemas, Universitaria Tecnológica de Santander).

Se trabaja con el dataset [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
(UCI id 374, licencia CC BY 4.0): 19 735 registros cada 10 minutos (ene–may 2016) de una vivienda de bajo consumo en
Stambruges, Bélgica, con temperatura y humedad por habitación, clima exterior y el consumo de los electrodomésticos
(`Appliances`, Wh).

**Pregunta de investigación:** ¿en qué medida las variables ambientales interiores y exteriores, junto con la
información temporal, permiten estimar el consumo energético de los electrodomésticos y reconocer grupos de patrones
de uso?

## Estructura

```
docs/
  Semana_12/   Comprensión y análisis exploratorio (notebook + PDF del primer avance)
  Semana_13/   02_preparacion.ipynb, 03_modelo_supervisado.ipynb, informe, data/ y preprocesador.joblib
src/           Paquete Python del proyecto
```

## Ejecución

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.13.

```bash
uv sync
uv run jupyter lab
```

Los datos se descargan con `ucimlrepo` (`fetch_ucirepo(id=374)`, requiere internet); también hay una copia en
`docs/Semana_13/data/`.

## Resultados hasta ahora

- Variables excluidas: `rv1`/`rv2` (aleatorias), `lights` (fuga), `RH_6` (sensor con fallas desde abril) y `date` cruda.
- Transformaciones: `log1p` del objetivo, hora cíclica, one-hot de día de la semana, `StandardScaler` en un `Pipeline`.
- Split cronológico 80/20. **Ridge (alpha=1000): MAE 40,9 Wh** (−13 % frente a la base por hora), R² 0,105.
  Random Forest queda por debajo de la base (el periodo de prueba es más cálido y no extrapola).

## Checklist

### 1. Comprensión del problema
- [x] Pregunta investigable
- [x] Origen del dataset
- [x] Licencia / condiciones de uso
- [x] Número de filas y variables
- [x] Variable objetivo
- [x] Pertinencia del dataset

### 2. Exploración
- [x] Tipos de variables
- [x] Estadísticas descriptivas
- [x] Visualizaciones
- [x] Distribución de la variable objetivo
- [x] Nulos, duplicados y atípicos

### 3. Preparación
- [x] Limpieza documentada
- [x] Imputación, codificación y escalado
- [x] Variables excluidas justificadas
- [x] Transformaciones ajustadas solo con entrenamiento (sin fuga)

### 4. Modelo supervisado
- [x] Variable objetivo y split cronológico
- [x] Entrenamiento de regresión
- [x] Métricas y justificación de parámetros
- [x] Predicciones

### 5. Modelo no supervisado
- [ ] Selección y escalado de variables
- [ ] Clustering (KMeans u otro justificado)
- [ ] Visualización de grupos
- [ ] Coherencia (silueta)

### 6. Análisis e interpretación
- [ ] Interpretar métricas
- [ ] Errores, sesgos y limitaciones
- [ ] Comparar ambos enfoques

### 7. Reproducibilidad
- [x] Datos y notebooks adjuntos
- [x] Instrucciones de ejecución
- [ ] Referencias completas
- [ ] Declaración de uso de IA

### Entregables
- [x] Semana 12: problema y análisis exploratorio
- [x] Semana 13: limpieza, transformación y primer modelo
- [ ] Semana 14: segundo modelo, evaluación, interpretación y video
- [ ] Semana 15: informe, archivos reproducibles y video
- [ ] Semana 16: sustentación

## Referencias

- Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). *Data driven prediction models of energy use of appliances in
  a low-energy house*. Energy and Buildings, 140, 81–97.
- Dataset: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction (CC BY 4.0)

# AMD Time-Series — RNN vs LSTM

Predicción del precio de cierre de **AMD (Advanced Micro Devices)** usando redes neuronales recurrentes. Compara 12 configuraciones entre **SimpleRNN** y **LSTM** variando la ventana de contexto (`time_step`) y el número de neuronas.

---

## Configuración en macOS y Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
source setup.sh
```

## Configuración en Windows

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate
setup.bat
```

Una vez configurado el entorno, ejecutar el pipeline:

```cmd
python main.py
```

---

## Descripción

El pipeline entrena y evalúa todas las combinaciones de arquitectura y configuración de forma automática:

| Arquitectura | time_step | Neuronas |
|---|---|---|
| SimpleRNN / LSTM | 5 | 5 |
| SimpleRNN / LSTM | 5 | 10 |
| SimpleRNN / LSTM | 10 | 10 |
| SimpleRNN / LSTM | 10 | 20 |
| SimpleRNN / LSTM | 15 | 15 |
| SimpleRNN / LSTM | 15 | 30 |

**Datos:** Precio de cierre diario de AMD 2020–2024 vía `yfinance`. Si no está disponible, se genera una serie sintética por Geometric Brownian Motion.  
**Métricas:** MAE (Mean Absolute Error) y R².  
**División:** 80% entrenamiento / 20% prueba, respetando orden cronológico.

---

## Estructura del proyecto

```
amd_timeseries_project/
│
├── src/
│   ├── data/
│   │   └── data_loader.py          # Descarga yfinance + fallback GBM
│   ├── features/
│   │   └── feature_engineering.py  # Split temporal, normalización, secuencias
│   ├── models/
│   │   ├── train.py                # Construcción y entrenamiento de modelos
│   │   └── evaluate.py             # Métricas y tabla comparativa
│   └── utils/
│       └── helpers.py              # Logging y visualizaciones
│
├── notebooks/
│   └── original_notebook.ipynb
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   └── test_evaluate.py
│
├── main.py                         # Punto de entrada
├── config.py                       # Configuración centralizada
├── requirements.txt
├── setup.py
├── setup.cfg
├── setup.sh                        # Script de instalación (macOS/Linux)
├── setup.bat                       # Script de instalación (Windows)
└── .env.example
```

---

## Resultados

Ranking final de los 12 modelos ordenado por MAE sobre datos de AMD 2020–2024:

| # | Modelo | time_step | Neuronas | MAE (USD) | R² | Épocas |
|---|---|---|---|---|---|---|
| 1 | SimpleRNN | 10 | 20 | 1.9715 | 0.9264 | 60 |
| 2 | SimpleRNN | 15 | 30 | 2.1122 | 0.9178 | 39 |
| 3 | LSTM | 15 | 30 | 2.3384 | 0.8936 | 60 |
| 4 | LSTM | 15 | 15 | 2.4183 | 0.8870 | 60 |
| 5 | LSTM | 10 | 20 | 2.5398 | 0.8733 | 60 |
| 6 | LSTM | 10 | 10 | 2.8859 | 0.8424 | 60 |
| 7 | SimpleRNN | 5 | 10 | 3.7330 | 0.7417 | 18 |
| 8 | SimpleRNN | 15 | 15 | 4.9949 | 0.3951 | 59 |
| 9 | SimpleRNN | 5 | 5 | 6.0930 | 0.2758 | 21 |
| 10 | LSTM | 5 | 5 | 9.1030 | -0.5998 | 11 |
| 11 | LSTM | 5 | 10 | 9.5380 | -0.7585 | 11 |
| 12 | SimpleRNN | 10 | 10 | 11.3977 | -1.1724 | 12 |

**Conclusiones principales:**

- El **mejor modelo fue SimpleRNN con ts=10 y 20 neuronas** (MAE=1.97 USD, R²=0.93), superando a todas las configuraciones LSTM.
- Las **ventanas cortas (ts=5) fueron perjudiciales** para ambas arquitecturas: los 4 peores modelos usan ts=5, con R² negativo en tres casos, lo que significa que predicen peor que usar el promedio histórico.
- **Más neuronas no siempre fue mejor:** SimpleRNN ts=15 con 30 neuronas superó a la misma arquitectura con 15 neuronas (MAE 2.11 vs 4.99), pero en otros casos la diferencia fue mínima.
- **LSTM fue más consistente en el rango medio:** los 4 modelos LSTM con ts=10 y ts=15 quedaron todos entre los puestos 3 y 6, sin caídas drásticas.
- La elección del `time_step` tuvo más impacto en el rendimiento que la arquitectura o el número de neuronas.

---

## Tests

```bash
pytest
```

---

## Configuración

Toda la configuración vive en `config.py`. Para cambiar el ticker, rango de fechas o hiperparámetros, edita directamente ese archivo. Para controlar el comportamiento en ejecución, usa las variables de entorno en `.env`:

| Variable | Valores | Efecto |
|---|---|---|
| `AMD_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` | Nivel de detalle en consola |
| `AMD_SHOW_PLOTS` | `0` / `1` | Mostrar gráficas en pantalla |
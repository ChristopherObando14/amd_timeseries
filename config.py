"""
config.py
=========
Configuración centralizada del proyecto AMD Time-Series.

Todas las constantes, rutas e hiperparámetros del pipeline se definen aquí.
Ningún módulo debe hardcodear valores; siempre debe importar desde este archivo.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

# ── Raíz del proyecto ─────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent

# ── Rutas de salida ───────────────────────────────────────────────────────────
FIGURES_DIR: Path = ROOT_DIR / "outputs" / "figures"
MODELS_DIR:  Path = ROOT_DIR / "outputs" / "models"
RESULTS_DIR: Path = ROOT_DIR / "outputs" / "results"


@dataclass(frozen=True)
class DataConfig:
    """Parámetros de descarga y partición de datos.

    Attributes:
        ticker: Símbolo bursátil a descargar.
        start: Fecha de inicio en formato 'YYYY-MM-DD'.
        end: Fecha de fin en formato 'YYYY-MM-DD'.
        interval: Frecuencia de las velas (yfinance).
        train_ratio: Fracción del dataset destinada al entrenamiento.
        seed: Semilla de reproducibilidad (también usada en fallback GBM).
    """
    ticker:      str   = "AMD"
    start:       str   = "2020-01-01"
    end:         str   = "2024-12-31"
    interval:    str   = "1d"
    train_ratio: float = 0.80
    seed:        int   = 1007272482


@dataclass(frozen=True)
class FeatureConfig:
    """Parámetros de ingeniería de features.

    Attributes:
        feature_range: Rango de normalización MinMaxScaler.
        configs: Lista de tuplas (time_step, units) a experimentar.
    """
    feature_range: Tuple[int, int]         = (0, 1)
    configs:       List[Tuple[int, int]]   = field(
        default_factory=lambda: [
            (5, 5), (5, 10),
            (10, 10), (10, 20),
            (15, 15), (15, 30),
        ]
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparámetros de entrenamiento.

    Attributes:
        model_types: Arquitecturas a comparar.
        epochs: Máximo de épocas por modelo.
        batch_size: Tamaño de batch.
        patience: Épocas sin mejora antes de EarlyStopping.
        validation_split: Fracción del training usada como validación interna.
        seed: Semilla para reproducibilidad de TensorFlow.
    """
    model_types:       List[str] = field(default_factory=lambda: ["SimpleRNN", "LSTM"])
    epochs:            int       = 60
    batch_size:        int       = 32
    patience:          int       = 10
    validation_split:  float     = 0.10
    seed:              int       = 1007272482


@dataclass(frozen=True)
class PlotConfig:
    """Configuración de visualizaciones.

    Attributes:
        dpi: Resolución de las figuras guardadas.
        figsize_wide: Tamaño para gráficos de comparativa.
        figsize_grid: Tamaño para grids 3x2 de predicciones/pérdidas.
        rnn_color: Color de línea para SimpleRNN.
        lstm_color: Color de línea para LSTM.
        real_color: Color de la serie real.
        train_color: Color del tramo de entrenamiento.
        test_color: Color del tramo de prueba.
    """
    dpi:          int           = 150
    figsize_wide: Tuple         = (16, 6)
    figsize_grid: Tuple         = (14, 12)
    rnn_color:    str           = "#1f77b4"
    lstm_color:   str           = "#d62728"
    real_color:   str           = "steelblue"
    pred_rnn:     str           = "tomato"
    pred_lstm:    str           = "darkorange"
    train_color:  str           = "steelblue"
    test_color:   str           = "tomato"


# ── Instancias globales listas para importar ──────────────────────────────────
DATA_CFG     = DataConfig()
FEATURE_CFG  = FeatureConfig()
TRAINING_CFG = TrainingConfig()
PLOT_CFG     = PlotConfig()

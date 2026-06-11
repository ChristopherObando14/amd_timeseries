"""
src/features/feature_engineering.py
====================================
Responsabilidades:
    1. Dividir el dataset respetando el orden cronológico (sin shuffle).
    2. Normalizar con MinMaxScaler ajustado *solo* en train (evita data leakage).
    3. Convertir series 1D en pares supervisados (X, y) mediante ventana deslizante.

Todas las funciones son puras: no modifican estado externo ni tienen efectos
secundarios. El scaler se devuelve explícitamente para reutilizarlo en
inferencia.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DataConfig, FeatureConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# División temporal
# ─────────────────────────────────────────────────────────────────────────────

def split_train_test(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el DataFrame en conjuntos de entrenamiento y prueba.

    La división se hace de forma cronológica (sin barajar) para preservar
    la estructura temporal de la serie.

    Args:
        df: DataFrame con índice DatetimeIndex y columna 'Close'.
        cfg: DataConfig con el ratio de entrenamiento (train_ratio).

    Returns:
        Tuple (train, test) donde cada elemento es un DataFrame independiente.

    Raises:
        ValueError: Si el ratio produce conjuntos vacíos.
    """
    split_idx = int(len(df) * cfg.train_ratio)

    if split_idx == 0 or split_idx >= len(df):
        raise ValueError(
            f"train_ratio={cfg.train_ratio} produce conjuntos vacíos "
            f"para un dataset de {len(df)} registros."
        )

    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()

    logger.info(
        "Split temporal — Train: %d (%s → %s) | Test: %d (%s → %s)",
        len(train), train.index[0].date(), train.index[-1].date(),
        len(test),  test.index[0].date(),  test.index[-1].date(),
    )
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Normalización (sin data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_scale(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: FeatureConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Ajusta MinMaxScaler en train y transforma ambos conjuntos.

    El scaler se ajusta *únicamente* con datos de entrenamiento para evitar
    que información del futuro (test) contamine el proceso de normalización.

    Args:
        train: DataFrame de entrenamiento con columna 'Close'.
        test: DataFrame de prueba con columna 'Close'.
        cfg: FeatureConfig con feature_range.

    Returns:
        Tuple (train_scaled, test_scaled, scaler) donde:
            - train_scaled: DataFrame de train con columna 'Close_scaled'.
            - test_scaled: DataFrame de test con columna 'Close_scaled'.
            - scaler: Instancia MinMaxScaler ya ajustada (para inverse_transform).
    """
    scaler = MinMaxScaler(feature_range=cfg.feature_range)

    train = train.copy()
    test  = test.copy()

    train["Close_scaled"] = scaler.fit_transform(train[["Close"]])
    test["Close_scaled"]  = scaler.transform(test[["Close"]])

    logger.info(
        "Normalización aplicada — rango: %s | min_train=%.2f | max_train=%.2f",
        cfg.feature_range,
        train["Close"].min(),
        train["Close"].max(),
    )
    return train, test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Creación de secuencias (ventana deslizante)
# ─────────────────────────────────────────────────────────────────────────────

def create_sequences(
    series: np.ndarray,
    time_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte una serie 1D en pares supervisados (X, y) por ventana deslizante.

    Para cada posición i genera:
        X[i] = series[i : i + time_step]   (contexto de longitud time_step)
        y[i] = series[i + time_step]        (valor objetivo)

    Args:
        series: Array 1D normalizado de longitud N.
        time_step: Longitud de la ventana de contexto.

    Returns:
        Tuple (X, y):
            - X: Array de forma (N - time_step, time_step).
            - y: Array de forma (N - time_step,).

    Raises:
        ValueError: Si time_step >= len(series).
    """
    if time_step >= len(series):
        raise ValueError(
            f"time_step ({time_step}) debe ser menor que la longitud de la serie ({len(series)})."
        )

    X, y = [], []
    for i in range(len(series) - time_step):
        X.append(series[i : i + time_step])
        y.append(series[i + time_step])

    return np.array(X), np.array(y)


def build_train_test_sequences(
    train: pd.DataFrame,
    test: pd.DataFrame,
    time_step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genera secuencias de entrenamiento y prueba para un time_step dado.

    Para el conjunto de prueba se concatenan los últimos `time_step` registros
    del entrenamiento, lo que permite generar el primer vector de contexto del
    test sin introducir data leakage.

    Args:
        train: DataFrame de train con columna 'Close_scaled'.
        test: DataFrame de test con columna 'Close_scaled'.
        time_step: Longitud de la ventana de contexto.

    Returns:
        Tuple (X_train, y_train, X_test, y_test) con:
            - X_train: (n_train, time_step, 1)
            - y_train: (n_train,)
            - X_test:  (n_test, time_step, 1)
            - y_test:  (n_test,)
    """
    train_scaled = train["Close_scaled"].values
    test_scaled  = test["Close_scaled"].values

    X_train, y_train = create_sequences(train_scaled, time_step)
    X_train = X_train.reshape(-1, time_step, 1)

    # Concatenar cola del train para el primer contexto del test
    combined = np.concatenate([train_scaled[-time_step:], test_scaled])
    X_test, y_test = create_sequences(combined, time_step)
    X_test = X_test.reshape(-1, time_step, 1)

    logger.debug(
        "Secuencias ts=%d — X_train:%s  X_test:%s",
        time_step, X_train.shape, X_test.shape,
    )
    return X_train, y_train, X_test, y_test

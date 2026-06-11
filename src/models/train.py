"""
src/models/train.py
===================
Responsabilidades:
    1. Construir modelos SimpleRNN / LSTM con Keras.
    2. Ejecutar el loop de entrenamiento sobre todas las configuraciones.
    3. Devolver resultados y artefactos de entrenamiento de forma estructurada.

Diseño:
    - Sin variables globales: todo fluye por parámetros.
    - Reproducibilidad garantizada fijando la semilla de TF antes de cada modelo.
    - Los modelos *no* se guardan en disco aquí; esa responsabilidad recae en helpers.py.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential

from config import TrainingConfig, FeatureConfig
from src.features.feature_engineering import build_train_test_sequences

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tipos de datos de retorno
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Contenedor de resultados de un modelo entrenado.

    Attributes:
        key: Identificador único del modelo (ej. 'LSTM_ts10_u20').
        model_type: 'SimpleRNN' | 'LSTM'.
        time_step: Longitud de la ventana de contexto.
        units: Número de neuronas en la capa recurrente.
        history: Diccionario con curvas de pérdida {'loss', 'val_loss'}.
        y_pred: Predicciones en escala original (USD).
        y_real: Valores reales en escala original (USD).
        dates: Índice temporal correspondiente a y_real / y_pred.
        n_epochs: Número de épocas efectivamente entrenadas.
    """
    key:        str
    model_type: str
    time_step:  int
    units:      int
    history:    Dict[str, List[float]]
    y_pred:     np.ndarray
    y_real:     np.ndarray
    dates:      pd.DatetimeIndex
    n_epochs:   int


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de modelos
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_type: str,
    time_step:  int,
    units:      int,
) -> Sequential:
    """Construye y compila un modelo SimpleRNN o LSTM de una sola capa recurrente.

    Arquitectura intencionalmente mínima para aislar el efecto de time_step
    y units sobre el rendimiento, sin que una arquitectura compleja enmascare
    las diferencias.

    Args:
        model_type: Tipo de capa recurrente: 'SimpleRNN' | 'LSTM'.
        time_step: Longitud de las secuencias de entrada.
        units: Número de neuronas en la capa recurrente.

    Returns:
        Modelo Keras compilado (Adam, pérdida MSE).

    Raises:
        ValueError: Si model_type no es 'SimpleRNN' ni 'LSTM'.
    """
    model = Sequential(name=f"{model_type}_ts{time_step}_u{units}")

    if model_type == "SimpleRNN":
        # activation='tanh' es estándar; evita ReLU por riesgo de exploding gradients.
        model.add(SimpleRNN(units, activation="tanh", input_shape=(time_step, 1)))
    elif model_type == "LSTM":
        model.add(LSTM(units, input_shape=(time_step, 1)))
    else:
        raise ValueError(
            f"model_type='{model_type}' no reconocido. Usa 'SimpleRNN' o 'LSTM'."
        )

    model.add(Dense(1))  # Salida escalar: precio del siguiente día
    model.compile(optimizer="adam", loss="mse")
    logger.debug("Modelo construido: %s", model.name)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Loop de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def train_single_model(
    model_type:   str,
    time_step:    int,
    units:        int,
    train:        pd.DataFrame,
    test:         pd.DataFrame,
    scaler:       MinMaxScaler,
    training_cfg: TrainingConfig,
) -> ModelResult:
    """Entrena un modelo individual y devuelve sus resultados completos.

    Args:
        model_type: 'SimpleRNN' | 'LSTM'.
        time_step: Longitud de ventana de contexto.
        units: Neuronas en la capa recurrente.
        train: DataFrame de train con columna 'Close_scaled'.
        test: DataFrame de test con columna 'Close_scaled'.
        scaler: Scaler ajustado en train (para inverse_transform).
        training_cfg: Hiperparámetros de entrenamiento.

    Returns:
        Instancia de ModelResult con predicciones, historial y metadatos.
    """
    key = f"{model_type}_ts{time_step}_u{units}"
    logger.info("Entrenando: %s", key)

    X_train, y_train, X_test, y_test = build_train_test_sequences(
        train, test, time_step
    )

    # Fijar semilla para reproducibilidad por modelo
    tf.random.set_seed(training_cfg.seed)

    model = build_model(model_type, time_step, units)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=training_cfg.patience,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_train, y_train,
        epochs=training_cfg.epochs,
        batch_size=training_cfg.batch_size,
        validation_split=training_cfg.validation_split,
        callbacks=[early_stop],
        verbose=0,
    )

    # Predicción e inversión de escala
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    n_epochs = len(history.history["loss"])
    logger.info("  %s — épocas: %d", key, n_epochs)

    return ModelResult(
        key=key,
        model_type=model_type,
        time_step=time_step,
        units=units,
        history=history.history,
        y_pred=y_pred,
        y_real=y_real,
        dates=test.index[-len(y_real):],
        n_epochs=n_epochs,
    )


def run_training_loop(
    train:        pd.DataFrame,
    test:         pd.DataFrame,
    scaler:       MinMaxScaler,
    feature_cfg:  FeatureConfig,
    training_cfg: TrainingConfig,
) -> List[ModelResult]:
    """Ejecuta el loop completo de entrenamiento sobre todas las configuraciones.

    Itera sobre cada combinación de (model_type, time_step, units) definida
    en los objetos de configuración y delega en train_single_model.

    Args:
        train: DataFrame de entrenamiento con 'Close_scaled'.
        test: DataFrame de prueba con 'Close_scaled'.
        scaler: MinMaxScaler ajustado (para inverse_transform).
        feature_cfg: Configuración de features (configs de ts/units).
        training_cfg: Configuración de entrenamiento (epochs, batch, etc.).

    Returns:
        Lista de ModelResult, uno por cada combinación entrenada.
    """
    results: List[ModelResult] = []
    total = len(training_cfg.model_types) * len(feature_cfg.configs)
    idx   = 0

    for model_type in training_cfg.model_types:
        logger.info("=" * 60)
        logger.info("  ARQUITECTURA: %s", model_type)
        logger.info("=" * 60)

        for time_step, units in feature_cfg.configs:
            idx += 1
            logger.info("[%d/%d] ts=%d  units=%d", idx, total, time_step, units)

            result = train_single_model(
                model_type=model_type,
                time_step=time_step,
                units=units,
                train=train,
                test=test,
                scaler=scaler,
                training_cfg=training_cfg,
            )
            results.append(result)

    logger.info("Loop de entrenamiento completado. Modelos: %d", len(results))
    return results

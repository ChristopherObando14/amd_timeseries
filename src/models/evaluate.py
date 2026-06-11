"""
src/models/evaluate.py
=======================
Responsabilidades:
    1. Calcular métricas (MAE, R²) para cada ModelResult.
    2. Construir y exportar la tabla comparativa de resultados.
    3. Identificar el modelo ganador según criterio seleccionable.

Ninguna función de este módulo produce visualizaciones; ese rol
pertenece a src/utils/helpers.py.
"""

import logging
from pathlib import Path
from typing import List, Literal

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.models.train import ModelResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo de métricas
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(result: ModelResult) -> dict:
    """Calcula MAE y R² para un ModelResult.

    Args:
        result: Instancia de ModelResult con y_real e y_pred.

    Returns:
        Diccionario con las siguientes claves:
            - 'Modelo': str
            - 'time_step': int
            - 'Neuronas': int
            - 'MAE': float (redondeado a 4 decimales)
            - 'R2': float (redondeado a 4 decimales)
            - 'Épocas': int
    """
    mae = mean_absolute_error(result.y_real, result.y_pred)
    r2  = r2_score(result.y_real, result.y_pred)

    return {
        "Modelo":     result.model_type,
        "time_step":  result.time_step,
        "Neuronas":   result.units,
        "MAE":        round(mae, 4),
        "R2":         round(r2, 4),
        "Épocas":     result.n_epochs,
    }


def build_results_dataframe(model_results: List[ModelResult]) -> pd.DataFrame:
    """Construye el DataFrame comparativo a partir de la lista de ModelResult.

    Args:
        model_results: Lista de ModelResult del loop de entrenamiento.

    Returns:
        DataFrame ordenado por MAE ascendente con todas las métricas.
    """
    rows = [compute_metrics(r) for r in model_results]
    df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    df.index += 1  # Ranking 1-based
    logger.info("Tabla comparativa construida. Modelos evaluados: %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Selección del mejor modelo
# ─────────────────────────────────────────────────────────────────────────────

def get_best_model(
    results_df: pd.DataFrame,
    criterion:  Literal["MAE", "R2"] = "MAE",
) -> pd.Series:
    """Identifica y retorna la fila del mejor modelo según la métrica indicada.

    Args:
        results_df: DataFrame producido por build_results_dataframe.
        criterion: Métrica de selección: 'MAE' (menor es mejor) o
                   'R2' (mayor es mejor). Por defecto 'MAE'.

    Returns:
        pd.Series con los datos del mejor modelo.

    Raises:
        ValueError: Si criterion no es 'MAE' ni 'R2'.
    """
    if criterion == "MAE":
        best = results_df.loc[results_df["MAE"].idxmin()]
    elif criterion == "R2":
        best = results_df.loc[results_df["R2"].idxmax()]
    else:
        raise ValueError(f"criterion='{criterion}' no válido. Usa 'MAE' o 'R2'.")

    logger.info(
        "Mejor modelo (%s): %s ts=%d u=%d — MAE=%.4f R²=%.4f",
        criterion,
        best["Modelo"], best["time_step"], best["Neuronas"],
        best["MAE"], best["R2"],
    )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Exportación de resultados
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Guarda la tabla de resultados como CSV en el directorio indicado.

    Args:
        results_df: DataFrame de resultados.
        output_dir: Directorio de destino (se crea si no existe).

    Returns:
        Path al archivo CSV generado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results_comparison.csv"
    results_df.to_csv(path)
    logger.info("Tabla de resultados guardada en: %s", path)
    return path


def print_results_table(results_df: pd.DataFrame) -> None:
    """Imprime la tabla comparativa formateada en consola.

    Args:
        results_df: DataFrame producido por build_results_dataframe.
    """
    separator = "=" * 62
    print(separator)
    print(f"{'TABLA COMPARATIVA — RNN vs LSTM':^62}")
    print(separator)
    print(results_df.to_string())
    print(separator)

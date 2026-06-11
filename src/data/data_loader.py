"""
src/data/data_loader.py
=======================
Responsabilidades:
    1. Descargar datos históricos de precios desde yfinance.
    2. Generar una serie sintética GBM si yfinance no está disponible.
    3. Exponer estadísticas descriptivas básicas (EDA).

Ninguna función de este módulo modifica estado global ni imprime
directamente a consola; los mensajes se devuelven o se loguean a través
del logger estándar de Python.
"""

import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from config import DataConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Descarga de datos reales
# ─────────────────────────────────────────────────────────────────────────────

def download_stock_data(cfg: DataConfig) -> pd.DataFrame:
    """Descarga el precio de cierre ajustado desde yfinance.

    Args:
        cfg: Instancia de DataConfig con ticker, start, end e interval.

    Returns:
        DataFrame con índice DatetimeIndex y columna 'Close', sin NaN.

    Raises:
        ImportError: Si yfinance no está instalado.
        ValueError: Si la descarga devuelve un DataFrame vacío.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance no está instalado. Ejecuta: pip install yfinance"
        ) from exc

    df_raw = yf.download(
        cfg.ticker,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        progress=False,
    )

    if df_raw.empty:
        raise ValueError(
            f"yfinance devolvió un DataFrame vacío para {cfg.ticker} "
            f"({cfg.start} → {cfg.end})."
        )

    df = df_raw[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)
    logger.info("Datos descargados desde yfinance. Shape: %s", df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: serie sintética por Geometric Brownian Motion
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_gbm(cfg: DataConfig) -> pd.DataFrame:
    """Genera una serie de precios sintética usando Geometric Brownian Motion.

    Se usa como fallback cuando yfinance no está disponible o falla.
    Los parámetros mu y sigma están calibrados para imitar el comportamiento
    histórico de AMD en el período 2020-2024.

    Args:
        cfg: Instancia de DataConfig. Se usan start, end y seed.

    Returns:
        DataFrame con índice DatetimeIndex (días hábiles) y columna 'Close'.
    """
    np.random.seed(cfg.seed)
    dates  = pd.bdate_range(start=cfg.start, end=cfg.end)
    price  = 50.0
    prices = []

    for _ in dates:
        price *= np.exp(np.random.normal(0.0008, 0.025))
        prices.append(price)

    prices_arr = np.array(prices)
    # Escalar a ~185 USD (máximo histórico del período)
    prices_arr = prices_arr * (185.0 / prices_arr.max())

    df = pd.DataFrame(
        {"Close": np.maximum(prices_arr, 5.0)},
        index=dates,
    )
    df.index.name = "Date"
    logger.warning("Usando serie sintética GBM (yfinance no disponible).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cargador principal con lógica de fallback
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg: DataConfig) -> Tuple[pd.DataFrame, str]:
    """Intenta descargar datos reales; cae en GBM sintético si falla.

    Args:
        cfg: Instancia de DataConfig.

    Returns:
        Tuple (df, source) donde:
            - df: DataFrame listo para el pipeline.
            - source: 'yfinance' | 'synthetic_gbm'.
    """
    try:
        df = download_stock_data(cfg)
        source = "yfinance"
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance no disponible (%s). Usando GBM sintético.", exc)
        df = generate_synthetic_gbm(cfg)
        source = "synthetic_gbm"

    logger.info(
        "Datos cargados — fuente: %s | shape: %s | rango: %s → %s",
        source,
        df.shape,
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df, source


# ─────────────────────────────────────────────────────────────────────────────
# Análisis Exploratorio de Datos (EDA)
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas descriptivas del DataFrame de precios.

    Args:
        df: DataFrame con columna 'Close'.

    Returns:
        DataFrame de estadísticas (output de describe() enriquecido con
        skewness y kurtosis).
    """
    stats = df.describe()
    stats.loc["skewness"] = df.skew()
    stats.loc["kurtosis"] = df.kurt()
    return stats


def validate_dataframe(df: pd.DataFrame) -> None:
    """Valida que el DataFrame cumpla los requisitos mínimos del pipeline.

    Args:
        df: DataFrame a validar.

    Raises:
        ValueError: Si el DataFrame está vacío, le falta la columna 'Close'
                    o contiene NaN en dicha columna.
    """
    if df.empty:
        raise ValueError("El DataFrame está vacío.")
    if "Close" not in df.columns:
        raise ValueError("El DataFrame debe tener una columna 'Close'.")
    if df["Close"].isna().any():
        raise ValueError("La columna 'Close' contiene valores NaN.")
    logger.debug("DataFrame validado correctamente (%d filas).", len(df))

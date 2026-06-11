"""
src/utils/helpers.py
=====================
Responsabilidades:
    1. Configurar logging del proyecto.
    2. Crear y guardar todas las visualizaciones del pipeline.
    3. Funciones de I/O auxiliares (crear directorios, suprimir warnings).

Cada función de visualización recibe todos sus datos como parámetros
y guarda la figura en la ruta indicada. No muestra ventanas interactivas
(plt.show() está deshabilitado en producción; se activa solo si
show=True se pasa explícitamente).
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DataConfig, PlotConfig
from src.models.train import ModelResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuración de logging y entorno
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configura el logger raíz del proyecto con formato estándar.

    Args:
        level: Nivel de logging (logging.DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def suppress_tf_warnings() -> None:
    """Silencia logs de TensorFlow y advertencias de Python.

    Establece TF_CPP_MIN_LOG_LEVEL=3 y filtra FutureWarnings de numpy/pandas.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def ensure_dirs(*dirs: Path) -> None:
    """Crea los directorios indicados si no existen.

    Args:
        *dirs: Rutas de directorios a crear.
    """
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug("Directorio asegurado: %s", d)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 0 — EDA: serie temporal + distribución
# ─────────────────────────────────────────────────────────────────────────────

def plot_eda(
    df:       pd.DataFrame,
    cfg_data: DataConfig,
    cfg_plot: PlotConfig,
    out_dir:  Path,
    show:     bool = False,
) -> Path:
    """Genera la figura de análisis exploratorio (serie + histograma).

    Args:
        df: DataFrame con índice DatetimeIndex y columna 'Close'.
        cfg_data: DataConfig (ticker, start, end).
        cfg_plot: PlotConfig (colores, dpi, tamaños).
        out_dir: Directorio donde se guardará la figura.
        show: Si True, muestra la figura en pantalla (modo notebook).

    Returns:
        Path al archivo PNG generado.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(df.index, df["Close"], color=cfg_plot.real_color, lw=1.2)
    axes[0].set_title(
        f"Precio de Cierre {cfg_data.ticker} "
        f"({cfg_data.start[:4]}–{cfg_data.end[:4]})"
    )
    axes[0].set_xlabel("Fecha")
    axes[0].set_ylabel("Precio (USD)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(df["Close"], bins=40, color=cfg_plot.real_color, edgecolor="white", alpha=0.8)
    axes[1].set_title("Distribución del Precio de Cierre")
    axes[1].set_xlabel("Precio (USD)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = out_dir / "fig0_eda.png"
    plt.savefig(out_path, dpi=cfg_plot.dpi)
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Figura guardada: %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — División temporal
# ─────────────────────────────────────────────────────────────────────────────

def plot_train_test_split(
    train:    pd.DataFrame,
    test:     pd.DataFrame,
    cfg_data: DataConfig,
    cfg_plot: PlotConfig,
    out_dir:  Path,
    show:     bool = False,
) -> Path:
    """Visualiza la división temporal train/test sobre la serie completa.

    Args:
        train: DataFrame de entrenamiento con columna 'Close'.
        test: DataFrame de prueba con columna 'Close'.
        cfg_data: DataConfig (ticker).
        cfg_plot: PlotConfig.
        out_dir: Directorio de salida.
        show: Si True, muestra la figura.

    Returns:
        Path al archivo PNG generado.
    """
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(train.index, train["Close"], color=cfg_plot.train_color, lw=1.3, label="Entrenamiento")
    ax.plot(test.index,  test["Close"],  color=cfg_plot.test_color,  lw=1.3, label="Prueba")
    ax.axvline(x=test.index[0], color="gray", ls="--", alpha=0.7, label="Corte temporal")
    ax.set_title(f"División Temporal {cfg_data.ticker} — 80% Train / 20% Test", fontsize=13)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (USD)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "fig1_division_temporal.png"
    plt.savefig(out_path, dpi=cfg_plot.dpi)
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Figura guardada: %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos para grids de predicciones y pérdidas
# ─────────────────────────────────────────────────────────────────────────────

def _get_results_by_type(
    model_results: List[ModelResult],
    model_type:    str,
) -> List[ModelResult]:
    """Filtra y ordena los ModelResult por tipo de arquitectura."""
    return [r for r in model_results if r.model_type == model_type]


def _build_results_index(model_results: List[ModelResult]) -> Dict[str, ModelResult]:
    """Crea un índice {key: ModelResult} para búsqueda rápida."""
    return {r.key: r for r in model_results}


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 & 3 — Predicciones (SimpleRNN y LSTM)
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    model_results: List[ModelResult],
    results_df:    pd.DataFrame,
    model_type:    str,
    configs:       List[Tuple[int, int]],
    cfg_plot:      PlotConfig,
    out_dir:       Path,
    show:          bool = False,
) -> Path:
    """Genera el grid 3×2 de predicciones vs valores reales.

    Args:
        model_results: Lista de todos los ModelResult del pipeline.
        results_df: DataFrame de métricas (MAE, R²).
        model_type: 'SimpleRNN' | 'LSTM'.
        configs: Lista de tuplas (time_step, units).
        cfg_plot: PlotConfig.
        out_dir: Directorio de salida.
        show: Si True, muestra la figura.

    Returns:
        Path al archivo PNG generado.
    """
    pred_color = cfg_plot.pred_rnn if model_type == "SimpleRNN" else cfg_plot.pred_lstm
    fig_num    = "2" if model_type == "SimpleRNN" else "3"
    arch_label = "RNN" if model_type == "SimpleRNN" else "LSTM"

    results_idx = _build_results_index(model_results)
    fig, axes = plt.subplots(3, 2, figsize=cfg_plot.figsize_grid)
    fig.suptitle(
        f"Predicciones {arch_label} — Valores Reales vs Predichos",
        fontsize=14, fontweight="bold",
    )

    for idx, (ts, u) in enumerate(configs):
        ax  = axes[idx // 2][idx % 2]
        key = f"{model_type}_ts{ts}_u{u}"
        r   = results_idx[key]

        mask = (
            (results_df["Modelo"]    == model_type) &
            (results_df["time_step"] == ts) &
            (results_df["Neuronas"]  == u)
        )
        row = results_df[mask].iloc[0]

        ax.plot(r.dates, r.y_real, color=cfg_plot.real_color, lw=1.2, label="Real",  alpha=0.9)
        ax.plot(r.dates, r.y_pred, color=pred_color,          lw=1.2, label="Pred",  alpha=0.85, ls="--")
        ax.set_title(f"ts={ts}, neuronas={u}  |  MAE={row.MAE:.2f}  |  R²={row.R2:.3f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylabel("Precio (USD)")

    plt.tight_layout()
    out_path = out_dir / f"fig{fig_num}_pred_{model_type.lower()}.png"
    plt.savefig(out_path, dpi=cfg_plot.dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Figura guardada: %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 & 5 — Curvas de pérdida (SimpleRNN y LSTM)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(
    model_results: List[ModelResult],
    model_type:    str,
    configs:       List[Tuple[int, int]],
    cfg_plot:      PlotConfig,
    out_dir:       Path,
    show:          bool = False,
) -> Path:
    """Genera el grid 3×2 de curvas de pérdida (train + val MSE).

    Args:
        model_results: Lista de ModelResult.
        model_type: 'SimpleRNN' | 'LSTM'.
        configs: Lista de tuplas (time_step, units).
        cfg_plot: PlotConfig.
        out_dir: Directorio de salida.
        show: Si True, muestra la figura.

    Returns:
        Path al archivo PNG generado.
    """
    val_color = cfg_plot.pred_rnn if model_type == "SimpleRNN" else cfg_plot.pred_lstm
    fig_num   = "4" if model_type == "SimpleRNN" else "5"
    arch_label = "RNN" if model_type == "SimpleRNN" else "LSTM"

    results_idx = _build_results_index(model_results)
    fig, axes = plt.subplots(3, 2, figsize=cfg_plot.figsize_grid)
    fig.suptitle(
        f"Curvas de Pérdida (MSE) — {arch_label}",
        fontsize=14, fontweight="bold",
    )

    for idx, (ts, u) in enumerate(configs):
        ax  = axes[idx // 2][idx % 2]
        key = f"{model_type}_ts{ts}_u{u}"
        h   = results_idx[key].history

        ax.plot(h["loss"],     color=cfg_plot.real_color, lw=1.5, label="Train Loss")
        ax.plot(h["val_loss"], color=val_color,           lw=1.5, label="Val Loss", ls="--")
        ax.set_title(f"{arch_label}  ts={ts}, neuronas={u}", fontsize=10)
        ax.set_xlabel("Época")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"fig{fig_num}_loss_{model_type.lower()}.png"
    plt.savefig(out_path, dpi=cfg_plot.dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Figura guardada: %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Comparativa final MAE / R²
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_comparison(
    results_df: pd.DataFrame,
    cfg_plot:   PlotConfig,
    out_dir:    Path,
    show:       bool = False,
) -> Path:
    """Genera el gráfico de barras comparativo (MAE y R²) de todos los modelos.

    Args:
        results_df: DataFrame con columnas Modelo, time_step, Neuronas, MAE, R2.
        cfg_plot: PlotConfig.
        out_dir: Directorio de salida.
        show: Si True, muestra la figura.

    Returns:
        Path al archivo PNG generado.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg_plot.figsize_wide)
    fig.suptitle("Comparativa Final de Todas las Configuraciones", fontsize=14, fontweight="bold")

    labels = [
        f"{r['Modelo']}\nts={r['time_step']}, u={r['Neuronas']}"
        for _, r in results_df.iterrows()
    ]
    colors = [
        cfg_plot.rnn_color if r["Modelo"] == "SimpleRNN" else cfg_plot.lstm_color
        for _, r in results_df.iterrows()
    ]
    x = np.arange(len(labels))

    # MAE
    bars1 = ax1.bar(x, results_df["MAE"], color=colors, width=0.6, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
    ax1.set_ylabel("MAE (USD)")
    ax1.set_title("MAE — menor es mejor")
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars1, results_df["MAE"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{v:.2f}", ha="center", va="bottom", fontsize=7,
        )

    # R²
    r2_vis = results_df["R2"].clip(lower=-1)
    bars2 = ax2.bar(x, r2_vis, color=colors, width=0.6, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
    ax2.set_ylabel("R²")
    ax2.set_title("R² — mayor es mejor")
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars2, results_df["R2"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0) + 0.02,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7,
        )

    leg_patches = [
        mpatches.Patch(color=cfg_plot.rnn_color,  label="SimpleRNN"),
        mpatches.Patch(color=cfg_plot.lstm_color,  label="LSTM"),
    ]
    ax1.legend(handles=leg_patches)
    ax2.legend(handles=leg_patches)

    plt.tight_layout()
    out_path = out_dir / "fig6_comparativa_final.png"
    plt.savefig(out_path, dpi=cfg_plot.dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    logger.info("Figura guardada: %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador de visualizaciones
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    df:            pd.DataFrame,
    train:         pd.DataFrame,
    test:          pd.DataFrame,
    model_results: List[ModelResult],
    results_df:    pd.DataFrame,
    configs:       List[Tuple[int, int]],
    cfg_data:      DataConfig,
    cfg_plot:      PlotConfig,
    figures_dir:   Path,
    show:          bool = False,
) -> List[Path]:
    """Genera y guarda todas las figuras del pipeline.

    Args:
        df: DataFrame completo (para EDA).
        train: DataFrame de entrenamiento.
        test: DataFrame de prueba.
        model_results: Lista de ModelResult.
        results_df: DataFrame de métricas.
        configs: Lista de (time_step, units).
        cfg_data: DataConfig.
        cfg_plot: PlotConfig.
        figures_dir: Directorio raíz de figuras.
        show: Si True, muestra cada figura.

    Returns:
        Lista de Paths a los archivos PNG generados.
    """
    ensure_dirs(figures_dir)
    paths = []

    paths.append(plot_eda(df, cfg_data, cfg_plot, figures_dir, show))
    paths.append(plot_train_test_split(train, test, cfg_data, cfg_plot, figures_dir, show))

    for arch in ["SimpleRNN", "LSTM"]:
        paths.append(
            plot_predictions(model_results, results_df, arch, configs, cfg_plot, figures_dir, show)
        )
        paths.append(
            plot_loss_curves(model_results, arch, configs, cfg_plot, figures_dir, show)
        )

    paths.append(plot_final_comparison(results_df, cfg_plot, figures_dir, show))

    logger.info("Todas las figuras generadas (%d archivos).", len(paths))
    return paths

"""
main.py
=======
Punto de entrada del pipeline AMD Time-Series.

Ejecuta el flujo completo en este orden:
    1. Configuración de entorno (logging, seeds, supresión de warnings)
    2. Ingesta de datos (yfinance o fallback GBM)
    3. Preprocesamiento (split temporal + normalización)
    4. Entrenamiento de los 12 modelos (6 SimpleRNN + 6 LSTM)
    5. Evaluación (MAE, R², tabla comparativa)
    6. Visualizaciones (7 figuras PNG en outputs/figures/)
    7. Exportación de resultados (CSV en outputs/results/)

Uso:
    python main.py

Variables de entorno opcionales (ver .env.example):
    AMD_LOG_LEVEL  : Nivel de logging (DEBUG|INFO|WARNING|ERROR). Default: INFO.
    AMD_SHOW_PLOTS : Mostrar gráficos en pantalla (0|1). Default: 0.
"""

import logging
import os
import sys

import numpy as np
import tensorflow as tf

# ── Añadir raíz del proyecto al path de Python ───────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_CFG,
    FEATURE_CFG,
    FIGURES_DIR,
    MODELS_DIR,
    PLOT_CFG,
    RESULTS_DIR,
    TRAINING_CFG,
)
from src.data.data_loader import (
    compute_descriptive_stats,
    load_data,
    validate_dataframe,
)
from src.features.feature_engineering import fit_and_scale, split_train_test
from src.models.evaluate import (
    build_results_dataframe,
    get_best_model,
    print_results_table,
    save_results_csv,
)
from src.models.train import run_training_loop
from src.utils.helpers import (
    ensure_dirs,
    generate_all_plots,
    setup_logging,
    suppress_tf_warnings,
)

# ─────────────────────────────────────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────────────────────────────────────

LOG_LEVEL  = os.getenv("AMD_LOG_LEVEL", "INFO").upper()
SHOW_PLOTS = os.getenv("AMD_SHOW_PLOTS", "0") == "1"

setup_logging(level=getattr(logging, LOG_LEVEL, logging.INFO))
suppress_tf_warnings()

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int) -> None:
    """Fija las semillas globales de NumPy y TensorFlow.

    Args:
        seed: Entero de semilla para reproducibilidad.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.debug("Seeds fijadas: numpy=%d, tensorflow=%d", seed, seed)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Orquesta el pipeline completo de entrenamiento y evaluación."""

    logger.info("=" * 65)
    logger.info("  AMD TIME-SERIES — RNN vs LSTM Pipeline")
    logger.info("=" * 65)

    # ── 0. Semillas y directorios ─────────────────────────────────────────────
    set_global_seeds(DATA_CFG.seed)
    ensure_dirs(FIGURES_DIR, MODELS_DIR, RESULTS_DIR)

    # ── 1. Ingesta de datos ───────────────────────────────────────────────────
    logger.info("[1/6] Cargando datos...")
    df, source = load_data(DATA_CFG)
    validate_dataframe(df)

    stats = compute_descriptive_stats(df)
    logger.info("Estadísticas:\n%s", stats.to_string())

    # ── 2. Preprocesamiento ───────────────────────────────────────────────────
    logger.info("[2/6] Preprocesando datos...")
    train, test = split_train_test(df, DATA_CFG)
    train_scaled, test_scaled, scaler = fit_and_scale(train, test, FEATURE_CFG)

    # ── 3. Entrenamiento ──────────────────────────────────────────────────────
    logger.info("[3/6] Entrenando modelos (%d configuraciones)...",
                len(TRAINING_CFG.model_types) * len(FEATURE_CFG.configs))

    model_results = run_training_loop(
        train=train_scaled,
        test=test_scaled,
        scaler=scaler,
        feature_cfg=FEATURE_CFG,
        training_cfg=TRAINING_CFG,
    )

    # ── 4. Evaluación ─────────────────────────────────────────────────────────
    logger.info("[4/6] Evaluando modelos...")
    results_df = build_results_dataframe(model_results)
    print_results_table(results_df)

    best = get_best_model(results_df, criterion="MAE")
    logger.info(
        "Mejor modelo: %s ts=%d u=%d — MAE=%.4f R²=%.4f",
        best["Modelo"], best["time_step"], best["Neuronas"],
        best["MAE"], best["R2"],
    )

    # ── 5. Exportar resultados ────────────────────────────────────────────────
    logger.info("[5/6] Guardando resultados...")
    csv_path = save_results_csv(results_df, RESULTS_DIR)
    logger.info("CSV guardado: %s", csv_path)

    # ── 6. Visualizaciones ────────────────────────────────────────────────────
    logger.info("[6/6] Generando visualizaciones...")
    generated_figs = generate_all_plots(
        df=df,
        train=train,
        test=test,
        model_results=model_results,
        results_df=results_df,
        configs=FEATURE_CFG.configs,
        cfg_data=DATA_CFG,
        cfg_plot=PLOT_CFG,
        figures_dir=FIGURES_DIR,
        show=SHOW_PLOTS,
    )
    logger.info("Figuras generadas: %d archivos en %s", len(generated_figs), FIGURES_DIR)

    # ── Resumen final ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  Pipeline completado exitosamente.")
    logger.info("  Fuente de datos : %s", source)
    logger.info("  Modelos         : %d", len(model_results))
    logger.info("  Resultados      : %s", csv_path)
    logger.info("  Figuras         : %s", FIGURES_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

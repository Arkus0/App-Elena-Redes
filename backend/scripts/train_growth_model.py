#!/usr/bin/env python3
"""
Growth Prediction Model Training Script

Script para entrenar/re-entrenar el modelo GrowthPredictionEngine periódicamente.
Puede ejecutarse como cron job o manualmente.

Uso:
    # Entrenar con datos sintéticos (para pruebas iniciales)
    python train_growth_model.py --synthetic --samples 1000

    # Entrenar con datos reales desde archivo CSV
    python train_growth_model.py --data-file training_data.csv

    # Entrenar con datos desde la base de datos
    python train_growth_model.py --from-db

    # Re-entrenar modelo existente con nuevos datos
    python train_growth_model.py --retrain --data-file new_data.csv

Ejemplo de cron job (re-entrenar diariamente a las 3 AM):
    0 3 * * * cd /path/to/backend && python scripts/train_growth_model.py --from-db

Autor: BrandPulse AI
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Añadir el directorio parent al path para importar módulos de la app
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from app.services.growth_prediction_engine import (
    GrowthPredictionEngine,
    GrowthPredictionConfig,
    get_growth_prediction_engine,
    reset_growth_engine,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


def load_data_from_csv(file_path: str) -> list:
    """
    Carga datos de entrenamiento desde un archivo CSV.

    El CSV debe tener las siguientes columnas:
    - posted_at (opcional): timestamp ISO
    - post_type: tipo de contenido (reel, carousel, static, etc.)
    - visual_energy: energía visual (0-1)
    - tempo: BPM del audio
    - brightness_variance: variación de brillo (0-1)
    - cut_density: cortes por minuto
    - sem_pca_1 a sem_pca_10: componentes PCA semánticos
    - rpi_score: variable objetivo (log-transformed RPI)

    Args:
        file_path: Ruta al archivo CSV

    Returns:
        Lista de diccionarios con los datos de entrenamiento
    """
    logger.info(f"Loading data from {file_path}")

    df = pd.read_csv(file_path)

    # Validar columnas requeridas
    required_columns = ["rpi_score"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convertir a lista de diccionarios
    data = df.to_dict(orient="records")

    logger.info(f"Loaded {len(data)} samples from CSV")
    return data


def load_data_from_database() -> list:
    """
    Carga datos de entrenamiento desde la base de datos.

    Esta función debe ser implementada según la estructura de tu base de datos.
    Por ahora, genera datos sintéticos como placeholder.

    Returns:
        Lista de diccionarios con los datos de entrenamiento
    """
    logger.info("Loading data from database...")

    # TODO: Implementar carga desde base de datos real
    # Ejemplo de cómo sería:
    #
    # from sqlalchemy import create_engine
    # from app.core.config import settings
    #
    # engine = create_engine(settings.DATABASE_URL)
    # query = """
    #     SELECT
    #         posted_at,
    #         post_type,
    #         visual_energy,
    #         tempo,
    #         brightness_variance,
    #         cut_density,
    #         sem_pca_1, sem_pca_2, ..., sem_pca_10,
    #         rpi_score
    #     FROM content_features
    #     WHERE rpi_score IS NOT NULL
    #     ORDER BY posted_at DESC
    #     LIMIT 10000
    # """
    # df = pd.read_sql(query, engine)
    # return df.to_dict(orient="records")

    logger.warning("Database connection not configured. Using synthetic data instead.")
    engine = GrowthPredictionEngine()
    return engine.generate_synthetic_training_data(n_samples=1000)


def validate_training_data(data: list) -> tuple:
    """
    Valida y limpia los datos de entrenamiento.

    Args:
        data: Lista de diccionarios con datos de entrenamiento

    Returns:
        Tupla (datos_limpios, estadísticas)
    """
    logger.info("Validating training data...")

    df = pd.DataFrame(data)
    initial_count = len(df)

    # Remover filas con rpi_score nulo o infinito
    df = df[df["rpi_score"].notna()]
    df = df[~df["rpi_score"].isin([np.inf, -np.inf])]

    # Clippear valores extremos de rpi_score
    df["rpi_score"] = df["rpi_score"].clip(lower=0, upper=5)

    # Llenar NaN en features numéricas con 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    final_count = len(df)
    removed = initial_count - final_count

    stats = {
        "initial_samples": initial_count,
        "final_samples": final_count,
        "removed_samples": removed,
        "rpi_score_mean": df["rpi_score"].mean(),
        "rpi_score_std": df["rpi_score"].std(),
        "rpi_score_min": df["rpi_score"].min(),
        "rpi_score_max": df["rpi_score"].max(),
    }

    logger.info(f"Validation complete: {final_count} samples kept, {removed} removed")
    logger.info(f"RPI score stats: mean={stats['rpi_score_mean']:.3f}, std={stats['rpi_score_std']:.3f}")

    return df.to_dict(orient="records"), stats


def train_model(
    data: list,
    config: GrowthPredictionConfig = None,
    save_model: bool = True
) -> dict:
    """
    Entrena el modelo con los datos proporcionados.

    Args:
        data: Lista de diccionarios con datos de entrenamiento
        config: Configuración opcional del modelo
        save_model: Si guardar el modelo después de entrenar

    Returns:
        Diccionario con métricas del entrenamiento
    """
    logger.info("Starting model training...")

    # Crear engine con configuración personalizada si se proporciona
    if config:
        engine = GrowthPredictionEngine(config=config)
    else:
        # Reiniciar singleton para asegurar modelo fresco
        reset_growth_engine()
        engine = get_growth_prediction_engine()

    # Entrenar
    metrics = engine.train(training_data=data, save_model=save_model)

    # Convertir métricas a diccionario
    metrics_dict = metrics.to_dict()

    logger.info("Training completed successfully!")
    logger.info(f"  - Train RMSE: {metrics.train_rmse:.4f}")
    logger.info(f"  - Test RMSE: {metrics.test_rmse:.4f}")
    logger.info(f"  - Test R²: {metrics.test_r2:.4f}")
    logger.info(f"  - CV RMSE: {metrics.cv_rmse_mean:.4f} (+/- {metrics.cv_rmse_std:.4f})")

    return metrics_dict


def save_training_report(metrics: dict, stats: dict, output_path: str = None):
    """
    Guarda un reporte de entrenamiento en formato JSON.

    Args:
        metrics: Métricas del modelo
        stats: Estadísticas de los datos
        output_path: Ruta de salida (opcional)
    """
    if output_path is None:
        output_path = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_statistics": stats,
        "model_metrics": metrics,
        "feature_importance_top5": dict(list(metrics.get("feature_importance", {}).items())[:5])
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Training report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train or retrain the Growth Prediction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with synthetic data
  python train_growth_model.py --synthetic --samples 1000

  # Train with real data from CSV
  python train_growth_model.py --data-file training_data.csv

  # Train with custom hyperparameters
  python train_growth_model.py --synthetic --n-estimators 300 --max-depth 8 --learning-rate 0.03
        """
    )

    # Fuente de datos
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for training"
    )
    data_group.add_argument(
        "--data-file",
        type=str,
        help="Path to CSV file with training data"
    )
    data_group.add_argument(
        "--from-db",
        action="store_true",
        help="Load training data from database"
    )

    # Configuración de datos sintéticos
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate (default: 1000)"
    )

    # Hiperparámetros del modelo
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of XGBoost estimators (default: 200)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: 6)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate / eta (default: 0.05)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )

    # Opciones de salida
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model to disk"
    )
    parser.add_argument(
        "--report-path",
        type=str,
        help="Path to save training report JSON"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Crear configuración personalizada si se especificaron hiperparámetros
    config = GrowthPredictionConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        cv_folds=args.cv_folds
    )

    try:
        # Cargar datos según la fuente especificada
        if args.synthetic:
            logger.info(f"Generating {args.samples} synthetic training samples...")
            engine = GrowthPredictionEngine()
            data = engine.generate_synthetic_training_data(n_samples=args.samples)
        elif args.data_file:
            data = load_data_from_csv(args.data_file)
        else:  # --from-db
            data = load_data_from_database()

        # Validar datos
        data, stats = validate_training_data(data)

        if len(data) < 100:
            logger.error(f"Insufficient training data: {len(data)} samples (minimum 100 required)")
            sys.exit(1)

        # Entrenar modelo
        metrics = train_model(
            data=data,
            config=config,
            save_model=not args.no_save
        )

        # Guardar reporte
        save_training_report(metrics, stats, args.report_path)

        # Resumen final
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Samples used:     {stats['final_samples']}")
        print(f"Test RMSE:        {metrics['test_rmse']:.4f}")
        print(f"Test R²:          {metrics['test_r2']:.4f}")
        print(f"CV RMSE:          {metrics['cv_rmse_mean']:.4f} (+/- {metrics['cv_rmse_std']:.4f})")
        print(f"Model version:    {metrics['model_version']}")
        print(f"Model saved:      {'No' if args.no_save else 'Yes'}")
        print("=" * 60)

        print("\nTop 5 Feature Importance:")
        for i, (feature, importance) in enumerate(list(metrics["feature_importance"].items())[:5], 1):
            print(f"  {i}. {feature}: {importance:.4f}")

        logger.info("Training script completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Fast Model Comparison - Strategic Selection
Only evaluate the most promising models based on financial ML best practices
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from loguru import logger
import warnings
import joblib
import json
warnings.filterwarnings('ignore')

from src.volatility_models import LightGBMModel, XGBoostModel, HARModel
from src.model_evaluator import VolatilityModelEvaluator


def main():
    """Streamlined model comparison"""
    logger.add("logs/fast_model_comparison.log", rotation="10 MB")

    logger.info("\n" + "="*80)
    logger.info("STRATEGIC VOLATILITY MODEL SELECTION - PHASE 2")
    logger.info("="*80 + "\n")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    logger.info("Loading processed data...")
    df = pd.read_parquet(config['data']['processed_file'])

    # Prepare data
    target_col = 'target_vol_24h'
    exclude_patterns = ['Open', 'High', 'Low', 'Close', 'Volume', 'target_', 'log_return']
    feature_cols = [col for col in df.columns if not any(p in col for p in exclude_patterns)]

    X = df[feature_cols].values
    y = df[target_col].values

    # Train/test split
    train_split = config['evaluation']['train_split']
    val_split = config['evaluation']['val_split']
    n = len(X)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    X_train_full = X[:val_end]
    y_train_full = y[:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]

    logger.info(f"Training: {len(X_train_full)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    logger.info(f"Features: {len(feature_cols)}")

    # Create only the best performing models
    logger.info("\nCreating strategic model selection...")
    models = {
        'LightGBM': LightGBMModel(**config['models']['volatility']['lightgbm']),
        'XGBoost': XGBoostModel(**config['models']['volatility']['xgboost']),
        'HAR': HARModel()
    }

    # Evaluate
    evaluator = VolatilityModelEvaluator(cv_splits=3)
    comparison_df = evaluator.compare_models(
        models, X_train_full, y_train_full, X_test, y_test, run_cv=False
    )

    # Display results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80 + "\n")
    print(comparison_df.to_string(index=False))

    # Save
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    evaluator.save_results(results_dir / 'detailed_results.json')

    # Best model
    best_model_name = evaluator.get_best_model('Test_RMSE')
    logger.info(f"\n{'='*80}")
    logger.info(f"RECOMMENDED MODEL: {best_model_name}")
    logger.info(f"{'='*80}\n")

    best_result = [r for r in evaluator.results if r['model_name'] == best_model_name][0]

    logger.info("Performance Metrics:")
    logger.info(f"  Test RMSE: {best_result['test_metrics']['RMSE']:.4f}")
    logger.info(f"  Test MAE: {best_result['test_metrics']['MAE']:.4f}")
    logger.info(f"  Test R²: {best_result['test_metrics']['R2']:.4f}")
    logger.info(f"  Test MAPE: {best_result['test_metrics']['MAPE']:.2f}%")
    logger.info(f"  Direction Accuracy: {best_result['test_metrics'].get('Direction_Accuracy', 0)*100:.2f}%")
    logger.info(f"  Overfitting: {best_result['overfitting']['severity']}")

    logger.info("\nRegime Performance:")
    for regime, metrics in best_result['regime_performance'].items():
        logger.info(f"  {regime.upper()}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")

    # Train and save best model
    logger.info(f"\nTraining final {best_model_name} on full data...")
    best_model = models[best_model_name]
    best_model.fit(X_train_full, y_train_full)

    model_path = Path('models') / f'volatility_model_{best_model_name.lower()}.pkl'
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'model_type': 'volatility_predictor',
        'features': feature_cols,
        'num_features': len(feature_cols),
        'target': target_col,
        'train_samples': len(X_train_full),
        'test_samples': len(X_test),
        'test_metrics': best_result['test_metrics'],
        'overfitting_analysis': best_result['overfitting'],
        'regime_performance': best_result['regime_performance']
    }

    with open(Path('models') / 'volatility_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("\n" + "="*80)
    logger.info("PHASE 2 COMPLETE - VOLATILITY MODEL READY")
    logger.info("="*80 + "\n")

    return comparison_df, best_model_name, best_model


if __name__ == "__main__":
    comparison_df, best_name, best_model = main()

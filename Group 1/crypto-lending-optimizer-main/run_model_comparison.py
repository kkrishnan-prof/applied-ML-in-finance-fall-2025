"""
Run Comprehensive Volatility Model Comparison
Trains and evaluates all volatility prediction models
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from src.volatility_models import (
    LightGBMModel, XGBoostModel, RandomForestModel,
    LSTMModel, HARModel, EnsembleModel
)
from src.model_evaluator import VolatilityModelEvaluator


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_processed_data(config):
    """Load processed data"""
    logger.info("Loading processed data...")
    df = pd.read_parquet(config['data']['processed_file'])
    logger.info(f"Loaded {len(df)} samples")
    return df


def prepare_data(df, config):
    """
    Prepare features and targets, split into train/val/test

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    logger.info("Preparing data for modeling...")

    # Get target column (24h volatility)
    target_col = 'target_vol_24h'

    # Get feature columns (exclude OHLCV and all targets)
    exclude_patterns = ['Open', 'High', 'Low', 'Close', 'Volume', 'target_', 'log_return']
    feature_cols = [
        col for col in df.columns
        if not any(pattern in col for pattern in exclude_patterns)
    ]

    logger.info(f"Using {len(feature_cols)} features")
    logger.info(f"Target: {target_col}")

    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values

    # Train/val/test split (chronological)
    train_split = config['evaluation']['train_split']
    val_split = config['evaluation']['val_split']

    n = len(X)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Val: {len(X_val)} samples")
    logger.info(f"Test: {len(X_test)} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def create_models(config, input_size):
    """
    Create all models to compare

    Returns:
        Dictionary of {model_name: model_instance}
    """
    logger.info("Creating models...")

    models = {}

    # 1. LightGBM (usually the best for tabular data)
    models['LightGBM'] = LightGBMModel(**config['models']['volatility']['lightgbm'])

    # 2. XGBoost
    models['XGBoost'] = XGBoostModel(**config['models']['volatility']['xgboost'])

    # 3. Random Forest (baseline)
    models['RandomForest'] = RandomForestModel(**config['models']['volatility']['randomforest'])

    # 4. LSTM (deep learning)
    # Note: LSTM is slower and may need GPU for good performance
    models['LSTM'] = LSTMModel(
        input_size=input_size,
        **config['models']['volatility']['lstm']
    )

    # 5. HAR (classical econometric model)
    models['HAR'] = HARModel()

    logger.info(f"Created {len(models)} models: {list(models.keys())}")

    return models


def main():
    """Main execution"""
    # Setup logging
    logger.add("logs/model_comparison.log", rotation="10 MB")

    logger.info("\n" + "="*80)
    logger.info("VOLATILITY MODEL COMPARISON - PHASE 2")
    logger.info("="*80 + "\n")

    # Load config
    config = load_config()

    # Load data
    df = load_processed_data(config)

    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_data(df, config)

    # Combine train + val for final training (as per time series best practice)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    logger.info(f"Combined training set: {len(X_train_full)} samples")

    # Create models
    models = create_models(config, input_size=len(feature_names))

    # Create evaluator
    evaluator = VolatilityModelEvaluator(cv_splits=3)  # Reduce CV splits for speed

    # Evaluate all models
    logger.info("\n" + "="*80)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("="*80 + "\n")

    # Skip CV to save time - we'll just do train/test
    comparison_df = evaluator.compare_models(
        models,
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        run_cv=False  # Set to True if you want CV (adds ~30 min)
    )

    # Display results
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("="*80 + "\n")

    print(comparison_df.to_string(index=False))

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    evaluator.save_results(results_dir / 'detailed_results.json')

    logger.info(f"\nResults saved to {results_dir}/")

    # Determine best model
    best_model_name = evaluator.get_best_model('Test_RMSE')
    logger.info(f"\n{'='*80}")
    logger.info(f"BEST MODEL: {best_model_name}")
    logger.info(f"{'='*80}\n")

    # Get best model results
    best_result = [r for r in evaluator.results if r['model_name'] == best_model_name][0]

    logger.info("\nBest Model Performance:")
    logger.info(f"  Test RMSE: {best_result['test_metrics']['RMSE']:.4f}")
    logger.info(f"  Test MAE: {best_result['test_metrics']['MAE']:.4f}")
    logger.info(f"  Test R²: {best_result['test_metrics']['R2']:.4f}")
    logger.info(f"  Test MAPE: {best_result['test_metrics']['MAPE']:.2f}%")
    logger.info(f"  Direction Accuracy: {best_result['test_metrics'].get('Direction_Accuracy', 0)*100:.2f}%")
    logger.info(f"  Overfitting: {best_result['overfitting']['severity']}")

    logger.info("\nPerformance by Volatility Regime:")
    for regime, metrics in best_result['regime_performance'].items():
        logger.info(f"  {regime.upper()}:")
        logger.info(f"    Samples: {metrics['sample_count']}")
        logger.info(f"    RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"    MAE: {metrics['MAE']:.4f}")
        logger.info(f"    R²: {metrics['R2']:.4f}")

    # Save best model
    best_model = models[best_model_name]
    best_model.fit(X_train_full, y_train_full)  # Retrain on full data

    import joblib
    model_path = Path('models') / f'best_model_{best_model_name.lower()}.pkl'
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info(f"\nBest model saved to {model_path}")

    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'features': feature_names,
        'num_features': len(feature_names),
        'train_samples': len(X_train_full),
        'test_samples': len(X_test),
        'test_metrics': best_result['test_metrics'],
        'overfitting': best_result['overfitting']
    }

    import json
    with open(Path('models') / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("\n" + "="*80)
    logger.info("PHASE 2 COMPLETE")
    logger.info("="*80 + "\n")

    return comparison_df, best_model_name


if __name__ == "__main__":
    comparison_df, best_model = main()

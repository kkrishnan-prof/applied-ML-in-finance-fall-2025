"""
Train Revenue Optimization Model - Phase 3
"""

import pandas as pd
import joblib
import json
from pathlib import Path
from loguru import logger
from src.revenue_optimizer import RevenueOptimizer


def main():
    logger.add("logs/revenue_model_training.log", rotation="10 MB")

    logger.info("\n" + "="*80)
    logger.info("PHASE 3: REVENUE OPTIMIZATION MODEL TRAINING")
    logger.info("="*80 + "\n")

    # Load processed data
    logger.info("Loading processed data...")
    df = pd.read_parquet('data/processed/processed_data.parquet')

    # Get market features (exclude targets and OHLCV)
    feature_cols = [
        col for col in df.columns
        if not any(x in col for x in ['target_', 'Open', 'High', 'Low', 'Close', 'Volume', 'log_return'])
    ]

    logger.info(f"Using {len(feature_cols)} features for revenue model")

    # Use training portion only (first 85%)
    train_end = int(len(df) * 0.85)
    market_features = df[feature_cols].iloc[:train_end]

    logger.info(f"Training samples: {len(market_features)}")

    # Initialize and train optimizer
    optimizer = RevenueOptimizer()
    optimizer.train(market_features)

    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / 'revenue_model.pkl'
    joblib.dump(optimizer, model_path)
    logger.info(f"\nRevenue model saved to {model_path}")

    # Save metadata
    metadata = {
        'model_type': 'revenue_optimizer',
        'features': feature_cols,
        'num_features': len(feature_cols),
        'train_samples': len(market_features),
        'simulation_params': optimizer.config['models']['revenue']['simulation'],
        'rate_range': {
            'min_daily': optimizer.config['models']['revenue']['simulation']['min_test_rate'],
            'max_daily': optimizer.config['models']['revenue']['simulation']['max_test_rate']
        }
    }

    metadata_path = model_dir / 'revenue_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {metadata_path}")

    # Test optimization on a few samples
    logger.info("\n" + "="*80)
    logger.info("TESTING RATE OPTIMIZATION")
    logger.info("="*80 + "\n")

    test_samples = market_features.iloc[::1000].head(5)  # Every 1000th sample, first 5

    for idx, row in test_samples.iterrows():
        result = optimizer.optimize_rate(row.values)

        logger.info(f"\nSample at index {idx}:")
        logger.info(f"  Market volatility: {row.get('realized_vol_24h', 0):.4f}")
        logger.info(f"  Volume ratio: {row.get('volume_ratio_24h', 1.0):.4f}")
        logger.info(f"  Optimal rate (daily): {result['optimal_rate_daily']*100:.4f}%")
        logger.info(f"  Expected revenue: {result['expected_revenue']:.6f}")

    logger.info("\n" + "="*80)
    logger.info("PHASE 3 COMPLETE - REVENUE MODEL READY")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()

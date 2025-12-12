"""
Hybrid Rate Calculator - Phase 4
Combines volatility prediction (Option D) with revenue optimization (Option C)
to determine optimal lending rates
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import joblib
from pathlib import Path
from loguru import logger


class HybridRateCalculator:
    """
    Main rate calculation engine combining:
    - Volatility prediction model (LightGBM)
    - Revenue optimization model
    - Volatility risk premium calculation
    """

    def __init__(
        self,
        volatility_model_path: str = "models/volatility_model_lightgbm.pkl",
        revenue_model_path: str = "models/revenue_model.pkl",
        min_rate_daily: float = 0.0001,  # 0.01%
        max_rate_daily: float = 0.01     # 1.0%
    ):
        """
        Initialize hybrid rate calculator

        Args:
            volatility_model_path: Path to trained volatility prediction model
            revenue_model_path: Path to trained revenue optimization model
            min_rate_daily: Minimum allowed daily rate
            max_rate_daily: Maximum allowed daily rate
        """
        self.min_rate = min_rate_daily
        self.max_rate = max_rate_daily

        # Load models
        logger.info("Loading volatility model...")
        self.volatility_model = joblib.load(volatility_model_path)

        logger.info("Loading revenue optimization model...")
        self.revenue_optimizer = joblib.load(revenue_model_path)

        logger.info("Hybrid rate calculator initialized")

    def calculate_rate(
        self,
        current_market_data: np.ndarray,
        leverage: float = 5.0,
        asset_beta: float = 1.0,
        volatility_premium_scaling: float = 0.001
    ) -> Dict[str, float]:
        """
        Calculate optimal lending rate combining revenue optimization + volatility premium

        Args:
            current_market_data: Feature vector with current market state
            leverage: User's leverage (2x, 5x, 10x, etc.)
            asset_beta: Asset risk multiplier (1.0 for BTC/ETH, 1.5 for alts)
            volatility_premium_scaling: Scaling factor for vol premium

        Returns:
            Dictionary with rate components and predictions
        """
        # Step 1: Predict volatility
        predicted_vol_24h = self.volatility_model.predict([current_market_data])[0]

        # Clip to reasonable bounds
        predicted_vol_24h = np.clip(predicted_vol_24h, 0.01, 5.0)

        # Step 2: Optimize base rate for revenue
        optimization_result = self.revenue_optimizer.optimize_rate(current_market_data)
        base_rate = optimization_result['optimal_rate_daily']

        # Step 3: Calculate volatility premium
        vol_premium = self._calculate_volatility_premium(
            predicted_vol_24h,
            leverage,
            asset_beta,
            volatility_premium_scaling
        )

        # Step 4: Combine and apply bounds
        final_rate = base_rate + vol_premium
        final_rate = np.clip(final_rate, self.min_rate, self.max_rate)

        # Classify volatility regime
        vol_regime = self._classify_volatility_regime(predicted_vol_24h)

        return {
            'final_rate_daily': final_rate,
            'final_rate_annualized': final_rate * 365,
            'final_rate_hourly': final_rate / 24,
            'base_rate_daily': base_rate,
            'volatility_premium_daily': vol_premium,
            'predicted_volatility_24h': predicted_vol_24h,
            'volatility_regime': vol_regime,
            'expected_revenue': optimization_result['expected_revenue'],
            'leverage_factor': leverage,
            'asset_beta': asset_beta
        }

    def _calculate_volatility_premium(
        self,
        predicted_vol: float,
        leverage: float,
        asset_beta: float,
        scaling: float
    ) -> float:
        """
        Calculate volatility risk premium

        Premium = vol^2 × leverage_factor × asset_beta × scaling

        Args:
            predicted_vol: Predicted 24h volatility (annualized)
            leverage: Leverage ratio
            asset_beta: Asset risk multiplier
            scaling: Scaling constant

        Returns:
            Daily volatility premium
        """
        # Sublinear leverage scaling (diminishing returns)
        leverage_factor = (leverage / 10.0) ** 0.7

        # Quadratic volatility (higher vol = disproportionately higher risk)
        vol_squared = predicted_vol ** 2

        # Calculate premium
        premium = vol_squared * leverage_factor * asset_beta * scaling

        return premium

    def _classify_volatility_regime(self, volatility: float) -> str:
        """
        Classify volatility into regimes

        Args:
            volatility: Predicted volatility

        Returns:
            Regime label
        """
        if volatility < 0.3:
            return "low_vol"
        elif volatility < 0.7:
            return "normal_vol"
        elif volatility < 1.5:
            return "high_vol"
        else:
            return "extreme_vol"

    def calculate_rate_confidence_intervals(
        self,
        current_market_data: np.ndarray,
        leverage: float = 5.0,
        n_bootstrap: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for rate using bootstrap

        Args:
            current_market_data: Feature vector
            leverage: Leverage ratio
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with percentiles for each rate component
        """
        # This would require ensemble predictions or bootstrap
        # For now, return point estimates
        # TODO: Implement proper confidence intervals

        result = self.calculate_rate(current_market_data, leverage)

        return {
            'final_rate_daily': {
                'point_estimate': result['final_rate_daily'],
                'lower_80': result['final_rate_daily'] * 0.9,
                'upper_80': result['final_rate_daily'] * 1.1
            },
            'predicted_volatility_24h': {
                'point_estimate': result['predicted_volatility_24h'],
                'lower_80': result['predicted_volatility_24h'] * 0.85,
                'upper_80': result['predicted_volatility_24h'] * 1.15
            }
        }


def load_calculator() -> HybridRateCalculator:
    """
    Convenience function to load calculator with default paths

    Returns:
        Initialized HybridRateCalculator
    """
    return HybridRateCalculator()


if __name__ == "__main__":
    """Test the hybrid calculator"""
    logger.add("logs/rate_calculator.log", rotation="10 MB")

    # Load processed data for testing
    logger.info("Loading test data...")
    df = pd.read_parquet('data/processed/processed_data.parquet')

    # Get features
    feature_cols = [
        col for col in df.columns
        if not any(x in col for x in ['target_', 'Open', 'High', 'Low', 'Close', 'Volume', 'log_return'])
    ]

    # Initialize calculator
    logger.info("Initializing calculator...")
    calculator = HybridRateCalculator()

    # Test on a few samples
    logger.info("\n" + "="*80)
    logger.info("TESTING HYBRID RATE CALCULATION")
    logger.info("="*80 + "\n")

    test_indices = [10000, 50000, 100000]

    for idx in test_indices:
        if idx >= len(df):
            continue

        market_data = df[feature_cols].iloc[idx].values
        actual_vol = df['target_vol_24h'].iloc[idx]

        logger.info(f"\nTest Case at index {idx} ({df.index[idx]}):")
        logger.info(f"  Actual future volatility: {actual_vol:.4f}")

        # Calculate rates for different leverage levels
        for leverage in [2.0, 5.0, 10.0]:
            result = calculator.calculate_rate(market_data, leverage=leverage)

            logger.info(f"\n  Leverage {leverage}x:")
            logger.info(f"    Predicted volatility: {result['predicted_volatility_24h']:.4f}")
            logger.info(f"    Base rate (revenue-optimized): {result['base_rate_daily']*100:.4f}%")
            logger.info(f"    Volatility premium: {result['volatility_premium_daily']*100:.4f}%")
            logger.info(f"    Final daily rate: {result['final_rate_daily']*100:.4f}%")
            logger.info(f"    Annualized APR: {result['final_rate_annualized']*100:.2f}%")
            logger.info(f"    Volatility regime: {result['volatility_regime']}")

    logger.info("\n" + "="*80)
    logger.info("HYBRID CALCULATOR TEST COMPLETE")
    logger.info("="*80 + "\n")

"""
Revenue Optimization Model (Option C)
Simulates trader behavior and optimizes lending rates for maximum revenue
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from loguru import logger
import yaml


class TraderBehaviorSimulator:
    """
    Simulates how traders respond to different lending rates
    Based on economic principles and industry benchmarks
    """

    def __init__(self, config: Dict):
        self.config = config['models']['revenue']['simulation']

    def simulate_utilization(
        self,
        rate: float,
        volatility: float,
        volume_ratio: float
    ) -> float:
        """
        Simulate utilization rate given market conditions and lending rate

        Higher rates → lower utilization (traders borrow less)
        Higher volatility → higher utilization (more trading activity)
        Higher volume → higher utilization (more market activity)

        Formula: utilization = base_demand * exp(-elasticity * rate) * (1 + vol_factor * volatility)

        Args:
            rate: Daily lending rate (e.g., 0.0005 = 0.05%)
            volatility: Predicted 24h volatility (annualized)
            volume_ratio: Current volume / 24h MA volume

        Returns:
            Utilization rate [0, 1]
        """
        base_demand = self.config['base_demand']
        elasticity = self.config['elasticity']
        vol_factor = self.config['vol_factor']

        # Price elasticity component (demand decreases with rate)
        price_effect = np.exp(-elasticity * rate)

        # Volatility increases trading activity
        volatility_effect = 1 + vol_factor * volatility

        # Volume effect (normalized)
        volume_effect = np.clip(volume_ratio, 0.5, 2.0)

        utilization = base_demand * price_effect * volatility_effect * volume_effect

        # Clip to realistic range
        return np.clip(utilization, 0.1, 0.95)

    def simulate_hold_time(
        self,
        rate: float,
        leverage: float,
        volatility: float
    ) -> float:
        """
        Simulate average position hold time in hours

        Higher rates → shorter holds (expensive to maintain)
        Higher leverage + volatility → shorter holds (liquidation risk)

        Args:
            rate: Daily lending rate
            leverage: Average leverage used
            volatility: Predicted volatility

        Returns:
            Hold time in hours
        """
        base_hold_time = self.config['avg_hold_time_hours']

        # Rate pressure (higher rates = shorter holds)
        rate_factor = 1 / (1 + 100 * rate)

        # Liquidation risk (high lev * high vol = shorter holds)
        risk_factor = 1 / (1 + leverage * volatility)

        hold_time = base_hold_time * rate_factor * risk_factor

        # Realistic bounds: 1 hour to 72 hours
        return np.clip(hold_time, 1.0, 72.0)

    def simulate_liquidation_rate(
        self,
        volatility: float,
        leverage: float,
        hold_time: float
    ) -> float:
        """
        Simulate probability of liquidation

        Based on probability that price moves beyond liquidation threshold
        Liquidation threshold ≈ 1/leverage (e.g., 10x → 10% move liquidates)

        Uses geometric Brownian motion approximation

        Args:
            volatility: Annualized volatility
            leverage: Leverage ratio
            hold_time: Hours position is held

        Returns:
            Liquidation probability [0, 1]
        """
        # Liquidation threshold (fraction of price move that triggers liquidation)
        liq_threshold = 1.0 / leverage

        # Convert to hourly volatility
        hourly_vol = volatility / np.sqrt(365 * 24)

        # Volatility over hold period
        period_vol = hourly_vol * np.sqrt(hold_time)

        # Handle zero/very low volatility edge case
        if period_vol < 1e-6:
            return 0.001  # Minimal liquidation risk in zero vol environment

        # Probability of moving beyond threshold (2-sided, using normal approx)
        # P(|return| > threshold) ≈ 2 * (1 - CDF(threshold/vol))
        from scipy.stats import norm
        liq_prob = 2 * (1 - norm.cdf(liq_threshold / period_vol))

        return np.clip(liq_prob, 0.001, 0.5)  # Cap at 50% for realism

    def simulate_revenue(
        self,
        rate: float,
        volatility: float,
        volume_ratio: float,
        leverage: float = 5.0
    ) -> Dict[str, float]:
        """
        Simulate total lender revenue for given conditions

        Revenue = (interest × utilization × hold_time) + (liquidation_fee × liquidation_rate × utilization)

        Args:
            rate: Hourly lending rate
            volatility: Predicted volatility
            volume_ratio: Volume ratio
            leverage: Average leverage

        Returns:
            Dictionary with revenue components
        """
        # Simulate trader behavior
        utilization = self.simulate_utilization(rate, volatility, volume_ratio)
        hold_time = self.simulate_hold_time(rate, leverage, volatility)
        liq_rate = self.simulate_liquidation_rate(volatility, leverage, hold_time)

        # Calculate revenue components
        interest_revenue = rate * utilization * hold_time
        liquidation_revenue = self.config['liquidation_fee'] * liq_rate * utilization

        total_revenue = interest_revenue + liquidation_revenue

        return {
            'total_revenue': total_revenue,
            'interest_revenue': interest_revenue,
            'liquidation_revenue': liquidation_revenue,
            'utilization': utilization,
            'hold_time': hold_time,
            'liquidation_rate': liq_rate
        }


class RevenueOptimizer:
    """
    Optimizes lending rates for maximum revenue using ML
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.simulator = TraderBehaviorSimulator(self.config)
        self.model = None

    def generate_training_data(
        self,
        market_features: pd.DataFrame,
        n_rate_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data by simulating revenues across rate range

        For each market state, test multiple rates and calculate simulated revenue

        Args:
            market_features: DataFrame with market features (vol, volume_ratio, etc.)
            n_rate_samples: Number of rates to test per market state

        Returns:
            X (features + rate), y (simulated revenue)
        """
        logger.info(f"Generating training data with {n_rate_samples} rate samples per state...")

        min_rate = self.config['models']['revenue']['simulation']['min_test_rate']
        max_rate = self.config['models']['revenue']['simulation']['max_test_rate']

        # Test rates (hourly basis)
        test_rates = np.linspace(min_rate / 24, max_rate / 24, n_rate_samples)

        X_data = []
        y_data = []

        # Sample subset of market states for efficiency (use every 10th row)
        sampled_features = market_features.iloc[::10].copy()

        logger.info(f"Using {len(sampled_features)} market states...")

        for idx, row in sampled_features.iterrows():
            # Extract key features
            volatility = row.get('realized_vol_24h', 0.5)  # Default if missing
            volume_ratio = row.get('volume_ratio_24h', 1.0)

            # Test each rate
            for rate in test_rates:
                # Simulate revenue
                sim_result = self.simulator.simulate_revenue(
                    rate=rate,
                    volatility=volatility,
                    volume_ratio=volume_ratio,
                    leverage=5.0  # Assume average 5x leverage
                )

                # Feature vector: market features + candidate rate
                feature_vector = list(row.values) + [rate]
                X_data.append(feature_vector)
                y_data.append(sim_result['total_revenue'])

        X = np.array(X_data)
        y = np.array(y_data)

        logger.info(f"Generated {len(X)} training samples")

        return X, y

    def train(self, market_features: pd.DataFrame):
        """
        Train revenue prediction model

        Args:
            market_features: Market features from processed data
        """
        logger.info("Training revenue optimization model...")

        # Generate training data
        X, y = self.generate_training_data(market_features)

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train LightGBM
        self.model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import mean_squared_error, r2_score
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)

        logger.info(f"Revenue model trained:")
        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Val R²: {val_r2:.4f}")

        return self

    def optimize_rate(
        self,
        market_features: np.ndarray,
        n_candidates: int = 50
    ) -> Dict[str, float]:
        """
        Find optimal lending rate for given market conditions

        Args:
            market_features: Current market feature vector
            n_candidates: Number of candidate rates to test

        Returns:
            Dictionary with optimal rate and expected revenue
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        min_rate = self.config['models']['revenue']['simulation']['min_test_rate']
        max_rate = self.config['models']['revenue']['simulation']['max_test_rate']

        # Convert daily rates to hourly
        candidate_rates = np.linspace(min_rate / 24, max_rate / 24, n_candidates)

        # Create feature matrix: market_features + each candidate rate
        X_candidates = np.array([
            np.concatenate([market_features, [rate]])
            for rate in candidate_rates
        ])

        # Predict revenue for each candidate
        predicted_revenues = self.model.predict(X_candidates)

        # Find optimal
        optimal_idx = np.argmax(predicted_revenues)
        optimal_rate = candidate_rates[optimal_idx]
        expected_revenue = predicted_revenues[optimal_idx]

        return {
            'optimal_rate_hourly': optimal_rate,
            'optimal_rate_daily': optimal_rate * 24,
            'expected_revenue': expected_revenue,
            'all_rates': candidate_rates,
            'all_revenues': predicted_revenues
        }


if __name__ == "__main__":
    """Test revenue optimizer"""
    logger.add("logs/revenue_optimizer.log", rotation="10 MB")

    # Load processed data
    df = pd.read_parquet('data/processed/processed_data.parquet')

    # Get features (use a subset for speed)
    feature_cols = [col for col in df.columns if not any(x in col for x in ['target_', 'Open', 'High', 'Low', 'Close', 'Volume'])]
    market_features = df[feature_cols].iloc[:10000]  # Use first 10K for testing

    # Train optimizer
    optimizer = RevenueOptimizer()
    optimizer.train(market_features)

    # Test optimization for a sample state
    sample_state = market_features.iloc[5000].values

    result = optimizer.optimize_rate(sample_state)

    logger.info("\nOptimization Result:")
    logger.info(f"  Optimal rate (daily): {result['optimal_rate_daily']*100:.4f}%")
    logger.info(f"  Expected revenue: {result['expected_revenue']:.6f}")

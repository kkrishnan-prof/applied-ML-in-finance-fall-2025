"""
Data Processing & Feature Engineering Module
Handles loading, validation, resampling, and feature generation for volatility prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
from loguru import logger


class DataProcessor:
    """
    Comprehensive data processing pipeline for crypto lending rate optimization
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config['data']
        self.feature_config = self.config['features']

    def load_and_validate_data(self) -> pd.DataFrame:
        """
        Load raw CSV data and perform validation

        Returns:
            Validated DataFrame with datetime index
        """
        logger.info(f"Loading data from {self.data_config['input_file']}")

        # Load CSV
        df = pd.read_csv(self.data_config['input_file'])

        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

        # Set as index and sort
        df.set_index('Timestamp', inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            # Forward fill missing values
            df.fillna(method='ffill', inplace=True)

        # Validate OHLC relationships
        invalid_ohlc = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        )

        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
            # Keep them but log

        # Check for zero/negative prices
        zero_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
        if zero_prices.any():
            logger.warning(f"Found {zero_prices.sum()} rows with zero/negative prices")
            # Remove these rows
            df = df[~zero_prices]

        return df

    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 1-minute data to 1-hour OHLCV candles

        Args:
            df: DataFrame with 1-minute data

        Returns:
            Resampled hourly DataFrame
        """
        logger.info("Resampling to hourly candles...")

        resampled = df.resample(self.data_config['resample_frequency']).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

        # Remove rows with NaN (incomplete candles)
        resampled.dropna(inplace=True)

        logger.info(f"Resampled to {len(resampled)} hourly candles")

        return resampled

    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate realized volatility and Parkinson volatility features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added volatility features
        """
        logger.info("Calculating volatility features...")

        # Calculate log returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Realized volatility for multiple windows
        for window in self.feature_config['volatility_windows']:
            # Annualized realized volatility
            df[f'realized_vol_{window}h'] = (
                df['log_return'].rolling(window).std() * np.sqrt(365 * 24)
            )

            # Parkinson volatility (uses high-low range)
            df[f'parkinson_vol_{window}h'] = (
                np.sqrt(
                    np.log(df['High'] / df['Low']).pow(2).rolling(window).mean() /
                    (4 * np.log(2))
                ) * np.sqrt(365 * 24)
            )

        # Volume volatility (coefficient of variation)
        for window in self.feature_config['volatility_windows']:
            vol_mean = df['Volume'].rolling(window).mean()
            vol_std = df['Volume'].rolling(window).std()
            df[f'volume_volatility_{window}h'] = vol_std / (vol_mean + 1e-10)

        return df

    def calculate_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate return and momentum features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added return features
        """
        logger.info("Calculating return features...")

        # Rolling returns for multiple windows
        for window in self.feature_config['return_windows']:
            df[f'return_{window}h'] = (
                (df['Close'] / df['Close'].shift(window)) - 1
            )

        # Return momentum indicators
        df['return_momentum_24h'] = df['log_return'].rolling(24).sum()
        df['return_momentum_72h'] = df['log_return'].rolling(72).sum()

        # Return acceleration
        df['return_acceleration'] = (
            df['return_24h'] - df['return_24h'].shift(24)
        )

        return df

    def calculate_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market regime indicators (trend, range, volume)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added regime features
        """
        logger.info("Calculating market regime features...")

        # Trend strength (simplified ADX)
        for window in [24, 72, 168]:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift(1))
            low_close = abs(df['Low'] - df['Close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            df[f'trend_strength_{window}h'] = (
                true_range.rolling(window).mean() / (df['Close'] + 1e-10)
            )

        # Range vs trending
        for window in [24, 72, 168]:
            price_range = df['High'].rolling(window).max() - df['Low'].rolling(window).min()
            price_std = df['Close'].rolling(window).std()
            df[f'range_vs_trend_{window}h'] = price_range / (price_std * np.sqrt(window) + 1e-10)

        # Volume profile
        for window in self.feature_config['volume_windows']:
            df[f'volume_ma_{window}h'] = df['Volume'].rolling(window).mean()
            df[f'volume_ratio_{window}h'] = (
                df['Volume'] / (df[f'volume_ma_{window}h'] + 1e-10)
            )

        # Price position in range
        for window in [24, 72, 168]:
            rolling_high = df['High'].rolling(window).max()
            rolling_low = df['Low'].rolling(window).min()
            df[f'price_position_{window}h'] = (
                (df['Close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
            )

        return df

    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time-based features with cyclical encoding

        Args:
            df: DataFrame with datetime index

        Returns:
            DataFrame with added time features
        """
        logger.info("Calculating time features...")

        # Hour of day
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Day of week
        df['day_of_week'] = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Week of year (seasonality)
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        return df

    def calculate_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate forward-looking target variables for volatility prediction

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with added target variables
        """
        logger.info("Calculating target variables...")

        # Forward realized volatility for multiple horizons
        # We calculate the rolling std of returns looking forward
        for horizon in self.feature_config['forward_vol_horizons']:
            if horizon == 1:
                # For 1-hour, use absolute return as proxy for volatility
                df[f'target_vol_{horizon}h'] = (
                    abs(np.log(df['Close'].shift(-1) / df['Close'])) * np.sqrt(365 * 24)
                )
            else:
                # Calculate forward log returns
                forward_returns = []
                for i in range(horizon):
                    forward_returns.append(
                        np.log(df['Close'].shift(-i-1) / df['Close'].shift(-i))
                    )

                # Stack returns and calculate volatility
                forward_returns_df = pd.concat(forward_returns, axis=1)
                df[f'target_vol_{horizon}h'] = (
                    forward_returns_df.std(axis=1) * np.sqrt(365 * 24)
                )

        # Extreme volatility flag (for classification task)
        vol_percentile = self.feature_config['extreme_vol_percentile']
        threshold = df['target_vol_24h'].quantile(vol_percentile / 100)
        df['target_extreme_vol_flag'] = (df['target_vol_24h'] > threshold).astype(int)

        # Volatility regime classification
        df['target_vol_regime'] = pd.cut(
            df['target_vol_24h'],
            bins=[0,
                  df['target_vol_24h'].quantile(0.25),
                  df['target_vol_24h'].quantile(0.75),
                  df['target_vol_24h'].max()],
            labels=['low_vol', 'normal_vol', 'high_vol']
        )

        return df

    def process_all(self) -> pd.DataFrame:
        """
        Execute complete data processing pipeline

        Returns:
            Fully processed DataFrame ready for modeling
        """
        logger.info("Starting complete data processing pipeline...")

        # Load and validate
        df = self.load_and_validate_data()

        # Resample to hourly
        df = self.resample_to_hourly(df)

        # Calculate all features
        df = self.calculate_volatility_features(df)
        df = self.calculate_return_features(df)
        df = self.calculate_market_regime_features(df)
        df = self.calculate_time_features(df)
        df = self.calculate_target_variables(df)

        # Remove rows with NaN in features (beginning due to rolling windows)
        # and targets (end due to forward-looking calculation)
        initial_rows = len(df)

        # Get feature columns (exclude OHLCV and targets)
        feature_cols = [col for col in df.columns
                       if not any(x in col for x in ['Open', 'High', 'Low', 'Close', 'Volume', 'target_', 'log_return'])]
        target_cols = [col for col in df.columns if 'target_vol' in col and col != 'target_vol_regime']

        # Drop rows where features or targets are NaN
        df = df.dropna(subset=feature_cols + target_cols)

        logger.info(f"Removed {initial_rows - len(df)} rows with NaN values ({initial_rows - len(df)} at start due to rolling windows, some at end due to forward targets)")

        # Save processed data
        output_path = self.data_config['processed_file']
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        logger.info(f"Saved processed data to {output_path}")

        # Summary statistics
        logger.info(f"\n=== PROCESSED DATA SUMMARY ===")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"Target variable range (24h vol): {df['target_vol_24h'].min():.4f} to {df['target_vol_24h'].max():.4f}")
        logger.info(f"Mean volatility: {df['target_vol_24h'].mean():.4f}")
        logger.info(f"Extreme volatility samples: {df['target_extreme_vol_flag'].sum()} ({df['target_extreme_vol_flag'].mean()*100:.2f}%)")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get categorized feature column names

        Returns:
            Dictionary with feature categories
        """
        all_cols = df.columns.tolist()

        # Exclude OHLCV and target columns
        exclude_patterns = ['Open', 'High', 'Low', 'Close', 'Volume', 'target_', 'log_return']

        feature_cols = [
            col for col in all_cols
            if not any(pattern in col for pattern in exclude_patterns)
        ]

        # Categorize features
        vol_features = [col for col in feature_cols if 'vol' in col.lower()]
        return_features = [col for col in feature_cols if 'return' in col or 'momentum' in col]
        regime_features = [col for col in feature_cols if any(x in col for x in ['trend', 'range', 'position'])]
        volume_features = [col for col in feature_cols if 'volume' in col]
        time_features = [col for col in feature_cols if any(x in col for x in ['hour', 'day', 'week', 'weekend'])]

        return {
            'all_features': feature_cols,
            'volatility_features': vol_features,
            'return_features': return_features,
            'regime_features': regime_features,
            'volume_features': volume_features,
            'time_features': time_features
        }


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/data_processing.log", rotation="10 MB")

    # Run processing
    processor = DataProcessor()
    df = processor.process_all()

    # Get feature columns
    features = processor.get_feature_columns(df)

    print("\n=== FEATURE SUMMARY ===")
    for category, cols in features.items():
        print(f"{category}: {len(cols)} features")
        if len(cols) <= 10:
            print(f"  {cols}")

    # Display sample
    print("\n=== SAMPLE DATA ===")
    print(df[features['all_features'][:10]].head())
    print("\n=== TARGET VARIABLES ===")
    print(df[[col for col in df.columns if 'target_' in col]].describe())

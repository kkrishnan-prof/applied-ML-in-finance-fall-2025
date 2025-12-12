"""
Volatility Model Implementations
Comprehensive collection of models for volatility forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
from loguru import logger


class BaseVolatilityModel:
    """Base class for all volatility models"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseVolatilityModel':
        """Train the model"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def get_params(self) -> Dict:
        """Get model parameters"""
        return {}


class LightGBMModel(BaseVolatilityModel):
    """LightGBM Regressor - typically best for tabular time series"""

    def __init__(self, **params):
        super().__init__("LightGBM")
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 7,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        default_params.update(params)
        self.model = LGBMRegressor(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMModel':
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_


class XGBoostModel(BaseVolatilityModel):
    """XGBoost Regressor - robust gradient boosting"""

    def __init__(self, **params):
        super().__init__("XGBoost")
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'  # Faster
        }
        default_params.update(params)
        self.model = XGBRegressor(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostModel':
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y, verbose=False)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_


class RandomForestModel(BaseVolatilityModel):
    """Random Forest Regressor - baseline ensemble model"""

    def __init__(self, **params):
        super().__init__("RandomForest")
        default_params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        self.model = RandomForestRegressor(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        logger.info(f"Training {self.name}...")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LSTMModel(BaseVolatilityModel):
    """LSTM Neural Network for time series"""

    def __init__(self, input_size: int, sequence_length: int = 24, **params):
        super().__init__("LSTM")
        self.input_size = input_size
        self.sequence_length = sequence_length

        # Hyperparameters
        self.hidden_size = params.get('hidden_units', 64)
        self.num_layers = params.get('num_layers', 2)
        self.dropout = params.get('dropout', 0.2)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.epochs = params.get('epochs', 50)
        self.batch_size = params.get('batch_size', 32)

        # Build model
        self.model = self._build_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _build_model(self) -> nn.Module:
        """Build LSTM architecture"""
        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Take last output
                out = lstm_out[:, -1, :]
                out = self.dropout(out)
                out = self.fc(out)
                return out.squeeze()

        return LSTMNet(self.input_size, self.hidden_size, self.num_layers, self.dropout)

    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """Create sequences for LSTM input"""
        sequences_X = []
        targets = []

        for i in range(len(X) - self.sequence_length):
            sequences_X.append(X[i:i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length])

        sequences_X = np.array(sequences_X)
        if y is not None:
            targets = np.array(targets)
            return sequences_X, targets
        return sequences_X

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSTMModel':
        logger.info(f"Training {self.name} (this may take a while)...")

        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False  # Don't shuffle time series!
        )

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()

        # Create sequences
        X_seq = self._create_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        # Pad to match original length (first sequence_length predictions are missing)
        preds = predictions.cpu().numpy()
        padded_preds = np.concatenate([
            np.full(self.sequence_length, preds[0]),  # Pad with first prediction
            preds
        ])

        return padded_preds


class HARModel(BaseVolatilityModel):
    """
    Heterogeneous Autoregressive (HAR) model
    Classic volatility forecasting model from Corsi (2009)
    """

    def __init__(self):
        super().__init__("HAR")
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()

    def _create_har_features(self, vol_series: np.ndarray) -> np.ndarray:
        """
        Create HAR features: daily, weekly, monthly realized volatility
        For hourly data: 24h, 168h (7 days), ~720h (30 days)
        """
        df = pd.DataFrame({'vol': vol_series})

        # Daily component (24 hours)
        df['vol_daily'] = df['vol'].rolling(24).mean()

        # Weekly component (168 hours)
        df['vol_weekly'] = df['vol'].rolling(168).mean()

        # Monthly component (720 hours)
        df['vol_monthly'] = df['vol'].rolling(720).mean()

        # Drop NaN
        df = df.dropna()

        return df[['vol_daily', 'vol_weekly', 'vol_monthly']].values

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HARModel':
        """
        For HAR, we need historical volatility, not all features
        X should be past realized volatility
        """
        logger.info(f"Training {self.name}...")

        # HAR uses its own features based on volatility
        # We'll use the first column as historical vol if X has multiple features
        if len(X.shape) > 1:
            historical_vol = X[:, 0]  # Assume first feature is volatility
        else:
            historical_vol = X

        # Create HAR features
        har_features = self._create_har_features(historical_vol)

        # Align targets (HAR features create NaN at start)
        y_aligned = y[-len(har_features):]

        self.model.fit(har_features, y_aligned)
        self.is_fitted = True
        self.last_vol = historical_vol  # Store for prediction
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using HAR model"""
        if len(X.shape) > 1:
            historical_vol = X[:, 0]
        else:
            historical_vol = X

        # Create HAR features
        har_features = self._create_har_features(historical_vol)

        # Predict
        predictions = self.model.predict(har_features)

        # Pad to match original length
        padded_preds = np.concatenate([
            np.full(len(X) - len(predictions), predictions[0]),
            predictions
        ])

        return padded_preds


class EnsembleModel(BaseVolatilityModel):
    """Weighted ensemble of multiple models"""

    def __init__(self, models: Dict[str, BaseVolatilityModel], weights: Dict[str, float] = None):
        super().__init__("Ensemble")
        self.models = models

        # Default to equal weights
        if weights is None:
            n = len(models)
            self.weights = {name: 1.0/n for name in models.keys()}
        else:
            self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        logger.info(f"Training ensemble of {len(self.models)} models...")

        for name, model in self.models.items():
            logger.info(f"Training ensemble component: {name}")
            model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        weights = []

        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.weights[name])

        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weighted_pred = np.average(predictions, axis=0, weights=weights)

        return weighted_pred


# Model factory
def create_model(model_type: str, **kwargs) -> BaseVolatilityModel:
    """Factory function to create models"""

    models = {
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'randomforest': RandomForestModel,
        'lstm': LSTMModel,
        'har': HARModel,
    }

    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type.lower()](**kwargs)

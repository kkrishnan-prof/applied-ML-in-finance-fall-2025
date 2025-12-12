"""
Model Evaluation Framework
Comprehensive evaluation and comparison of volatility models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger
import json


class VolatilityModelEvaluator:
    """
    Rigorous evaluation framework for volatility models
    """

    def __init__(self, cv_splits: int = 5):
        self.cv_splits = cv_splits
        self.results = {}

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics

        Returns:
            Dictionary with all evaluation metrics
        """
        # Filter out any NaN or inf values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            logger.warning("No valid predictions to evaluate")
            return {metric: np.nan for metric in [
                'RMSE', 'MAE', 'R2', 'MAPE', 'Mean_Error', 'Median_AE',
                'Max_Error', 'Relative_RMSE', 'Relative_MAE'
            ]}

        metrics = {
            # Standard regression metrics
            'RMSE': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'MAE': mean_absolute_error(y_true_clean, y_pred_clean),
            'R2': r2_score(y_true_clean, y_pred_clean),
            'MAPE': mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100,

            # Custom metrics
            'Mean_Error': np.mean(y_pred_clean - y_true_clean),
            'Median_AE': np.median(np.abs(y_pred_clean - y_true_clean)),
            'Max_Error': np.max(np.abs(y_pred_clean - y_true_clean)),

            # Relative metrics
            'Relative_RMSE': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)) / np.mean(y_true_clean),
            'Relative_MAE': mean_absolute_error(y_true_clean, y_pred_clean) / np.mean(y_true_clean),
        }

        # Direction accuracy (did we predict increase/decrease correctly?)
        if len(y_true_clean) > 1:
            y_true_prev = np.concatenate([[y_true_clean[0]], y_true_clean[:-1]])
            y_pred_prev = np.concatenate([[y_pred_clean[0]], y_pred_clean[:-1]])

            true_direction = np.sign(y_true_clean - y_true_prev)
            pred_direction = np.sign(y_pred_clean - y_pred_prev)

            metrics['Direction_Accuracy'] = np.mean(true_direction == pred_direction)
        else:
            metrics['Direction_Accuracy'] = np.nan

        return metrics

    def time_series_cv(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """
        Time series cross-validation with expanding window

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary of metric lists across folds
        """
        logger.info(f"Running {self.cv_splits}-fold time series CV for {model.name}")

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        cv_scores = {
            'RMSE': [], 'MAE': [], 'R2': [], 'MAPE': [],
            'Direction_Accuracy': [], 'Relative_RMSE': []
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"  Fold {fold+1}/{self.cv_splits}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred)

            for key in cv_scores.keys():
                if key in metrics:
                    cv_scores[key].append(metrics[key])

        # Average across folds
        cv_avg = {k: np.nanmean(v) for k, v in cv_scores.items()}
        cv_std = {k + '_std': np.nanstd(v) for k, v in cv_scores.items()}

        return {**cv_avg, **cv_std}

    def train_test_evaluation(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate model on train and test sets

        Returns:
            Tuple of (train_metrics, test_metrics)
        """
        logger.info(f"Training {model.name} on full training set")

        # Train
        model.fit(X_train, y_train)

        # Predict on both sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)

        return train_metrics, test_metrics

    def detect_overfitting(self, train_metrics: Dict, test_metrics: Dict) -> Dict:
        """
        Quantify overfitting severity

        Returns:
            Dictionary with overfitting analysis
        """
        gaps = {}

        # Calculate performance gaps
        for metric in ['RMSE', 'MAE', 'R2']:
            if metric in train_metrics and metric in test_metrics:
                if metric == 'R2':
                    # For R2, higher is better, so gap is train - test
                    gaps[f'{metric}_gap'] = train_metrics[metric] - test_metrics[metric]
                else:
                    # For error metrics, lower is better, so gap is (test - train) / train
                    gaps[f'{metric}_gap'] = (
                        (test_metrics[metric] - train_metrics[metric]) / (train_metrics[metric] + 1e-10)
                    )

        # Calculate overall overfitting score
        overfitting_score = np.mean([
            max(0, gaps.get('RMSE_gap', 0)),
            max(0, gaps.get('MAE_gap', 0)),
            max(0, gaps.get('R2_gap', 0))
        ])

        # Classify severity
        if overfitting_score < 0.1:
            severity = 'None'
        elif overfitting_score < 0.2:
            severity = 'Mild'
        elif overfitting_score < 0.4:
            severity = 'Moderate'
        else:
            severity = 'Severe'

        return {
            'score': overfitting_score,
            'severity': severity,
            'gaps': gaps
        }

    def evaluate_by_regime(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Performance breakdown by volatility regime

        Args:
            model: Fitted model
            X: Features
            y: True values
            y_pred: Predictions

        Returns:
            Metrics by regime
        """
        # Define regimes based on historical volatility percentiles
        vol_25 = np.percentile(y, 25)
        vol_75 = np.percentile(y, 75)

        regimes = {
            'low_vol': y <= vol_25,
            'normal_vol': (y > vol_25) & (y <= vol_75),
            'high_vol': y > vol_75
        }

        regime_metrics = {}
        for regime_name, mask in regimes.items():
            if np.sum(mask) > 0:
                regime_metrics[regime_name] = self.calculate_metrics(y[mask], y_pred[mask])
                regime_metrics[regime_name]['sample_count'] = int(np.sum(mask))

        return regime_metrics

    def evaluate_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_cv: bool = True
    ) -> Dict:
        """
        Comprehensive evaluation of a single model

        Returns:
            Complete evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model.name}")
        logger.info(f"{'='*60}")

        results = {
            'model_name': model.name,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        # Cross-validation
        if run_cv:
            cv_results = self.time_series_cv(model, X_train, y_train)
            results['cv_metrics'] = cv_results

        # Train/test evaluation
        train_metrics, test_metrics = self.train_test_evaluation(
            model, X_train, y_train, X_test, y_test
        )

        results['train_metrics'] = train_metrics
        results['test_metrics'] = test_metrics

        # Overfitting analysis
        overfitting = self.detect_overfitting(train_metrics, test_metrics)
        results['overfitting'] = overfitting

        # Get test predictions for regime analysis
        y_test_pred = model.predict(X_test)

        # Regime-specific performance
        regime_performance = self.evaluate_by_regime(model, X_test, y_test, y_test_pred)
        results['regime_performance'] = regime_performance

        # Log summary
        logger.info(f"\nTest Metrics:")
        logger.info(f"  RMSE: {test_metrics['RMSE']:.4f}")
        logger.info(f"  MAE: {test_metrics['MAE']:.4f}")
        logger.info(f"  R²: {test_metrics['R2']:.4f}")
        logger.info(f"  Direction Accuracy: {test_metrics.get('Direction_Accuracy', 0):.4f}")
        logger.info(f"\nOverfitting: {overfitting['severity']} (score: {overfitting['score']:.4f})")

        return results

    def compare_models(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_cv: bool = False  # CV is slow, skip by default
    ) -> pd.DataFrame:
        """
        Evaluate and compare multiple models

        Args:
            models: Dictionary of {model_name: model_instance}
            X_train, y_train: Training data
            X_test, y_test: Test data
            run_cv: Whether to run cross-validation (slow)

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model_name, model in models.items():
            try:
                result = self.evaluate_model(
                    model, X_train, y_train, X_test, y_test, run_cv=run_cv
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue

        # Store full results
        self.results = results

        # Create comparison dataframe
        comparison_data = []
        for result in results:
            row = {
                'Model': result['model_name'],
                'Test_RMSE': result['test_metrics']['RMSE'],
                'Test_MAE': result['test_metrics']['MAE'],
                'Test_R2': result['test_metrics']['R2'],
                'Test_MAPE': result['test_metrics']['MAPE'],
                'Direction_Accuracy': result['test_metrics'].get('Direction_Accuracy', np.nan),
                'Train_RMSE': result['train_metrics']['RMSE'],
                'Overfitting': result['overfitting']['severity'],
                'Overfitting_Score': result['overfitting']['score']
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by test RMSE
        df = df.sort_values('Test_RMSE')

        return df

    def save_results(self, filepath: str):
        """Save evaluation results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

    def get_best_model(self, metric: str = 'Test_RMSE') -> str:
        """Get name of best performing model"""
        if not self.results:
            return None

        if 'R2' in metric:
            # Higher is better
            best = max(self.results, key=lambda x: x['test_metrics'][metric.replace('Test_', '')])
        else:
            # Lower is better
            best = min(self.results, key=lambda x: x['test_metrics'][metric.replace('Test_', '')])

        return best['model_name']

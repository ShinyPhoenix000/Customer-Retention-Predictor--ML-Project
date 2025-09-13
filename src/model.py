"""
Model training and evaluation module for customer retention prediction.
Uses XGBoost classifier with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
from typing import Dict, Any
from pathlib import Path

class CustomerRetentionModel:
    """Class for training and evaluating the XGBoost model."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration.
        
        Args:
            config (Dict[str, Any]): Model configuration parameters
        """
        self.config = config
        params = self.config.get('params', {}).copy()
        
        # Set default eval_metric in params
        if 'eval_metric' not in params:
            params['eval_metric'] = ['logloss', 'error']
            
        self.model = XGBClassifier(**params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """
        Train the XGBoost model with optional validation set.
        """
        if X_val is not None and y_val is not None:
            print("Training with validation set...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)]
            )
        else:
            print("Training without validation set...")
            self.model.fit(X_train, y_train)
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Best parameters found
        """
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        grid_search = GridSearchCV(
            estimator=XGBClassifier(
                eval_metric=['logloss', 'error'],
                use_label_encoder=False
            ),
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        # If validation set is provided, use it in fit
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            grid_search.fit(X_train, y_train, 
                          eval_set=eval_set,
                          early_stopping_rounds=10,
                          verbose=False)
        else:
            grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
        """
        Perform hyperparameter tuning using GridSearchCV.
        """
        param_grid = self.config.get('param_grid', {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        })
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=self.config.get('cross_validation_folds', 5),
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data and return metrics.
        """
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else y_pred
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        return metrics
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model to disk.
        """
        joblib.dump(self.model, save_path)
        
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model file
        
        Raises:
            FileNotFoundError: If model file does not exist
        """
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not model_path.is_file():
            raise ValueError(f"Expected a file but got a directory: {model_path}")
            
        self.model = joblib.load(str(model_path))
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (pd.DataFrame): Features to make predictions on
            
        Returns:
            np.ndarray: Array of predictions (0 or 1)
        """
        return self.model.predict(X)
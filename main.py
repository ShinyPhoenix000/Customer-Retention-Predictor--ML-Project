"""
Customer Retention Predictor - Main Script

This script orchestrates the ML pipeline for predicting customer churn:
1. Loads and preprocesses customer data
2. Trains an XGBoost model
3. Evaluates model performance
4. Generates SHAP-based explanations
5. Makes predictions on new data

Usage:
    python main.py --config config/config.yaml --mode train
    python main.py --config config/config.yaml --mode predict
    python main.py --config config/config.yaml --mode explain
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler

from src.data_prep import DataPreparation
from src.model import CustomerRetentionModel
from src.explain import ModelExplainer
from src.utils import setup_logging, load_config, save_metrics

# Set up rich console for pretty printing
console = Console()
logger = logging.getLogger(__name__)
from sklearn.preprocessing import LabelEncoder


def find_latest_experiment(base_dir: str) -> Optional[Path]:
    """
    Find the most recent experiment directory.
    
    Args:
        base_dir (str): Base directory containing experiment folders
        
    Returns:
        Optional[Path]: Path to latest experiment directory, or None if no experiments exist
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
        
    experiments = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('experiment_')]
    if not experiments:
        return None
        
    latest_experiment = max(experiments, key=lambda x: x.stat().st_mtime)
    return latest_experiment

def setup_experiment_dir(base_dir: str) -> Path:
    """
    Create a timestamped experiment directory.
    
    Args:
        base_dir (str): Base directory for experiments
        
    Returns:
        Path: Path to created experiment directory
    """
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"experiment_{timestamp}"
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreated experiment directory: {exp_dir}")
    return exp_dir

def convert_target_to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert Yes/No target to 1/0.
    
    Args:
        series (pd.Series): Target column
        
    Returns:
        pd.Series: Numeric target column
    """
    if series.dtype == 'object':
        print(f"\nConverting target from Yes/No to 1/0")
        print(f"Original value counts:\n{series.value_counts()}")
        numeric_series = (series.str.lower() == 'yes').astype(int)
        print(f"Converted value counts:\n{numeric_series.value_counts()}")
        return numeric_series
    return series

def save_predictions(predictions: np.ndarray, output_path: Path, as_labels: bool = False) -> None:
    """
    Save predictions to a CSV file.
    
    Args:
        predictions (np.ndarray): Model predictions
        output_path (Path): Path to save predictions
        as_labels (bool): Whether to convert 0/1 to No/Yes
    """
    df = pd.DataFrame(predictions, columns=['prediction'])
    if as_labels:
        df['prediction'] = df['prediction'].map({0: 'No', 1: 'Yes'})
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    parser = argparse.ArgumentParser(
        description='Customer Retention Predictor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'predict', 'explain'],
        default='train', 
        help='Operation mode'
    )
    
    parser.add_argument(
        '--output-labels',
        action='store_true',
        help='Output predictions as Yes/No instead of 1/0'
    )
    
    args = parser.parse_args()

    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {args.config}")

    return args


def train_model(config: Dict[str, Any], data_prep: DataPreparation, 
              experiment_dir: Path) -> Tuple[CustomerRetentionModel, ModelExplainer]:
    """
    Train and evaluate the model, saving all artifacts to experiment directory.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        data_prep (DataPreparation): Initialized data preparation pipeline
        experiment_dir (Path): Directory to save outputs
        
    Returns:
        Tuple[CustomerRetentionModel, ModelExplainer]: Trained model and explainer
    """
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    raw_data = data_prep.load_data(config['data_path'])
    print(f"Loaded raw data shape: {raw_data.shape}")
    
    # Verify features from config exist in data
    feature_config = config['data_prep']['preprocessing']
    numerical_features = feature_config['numerical_features']
    categorical_features = feature_config['categorical_features']
    
    # Validate features exist in dataset
    all_features = numerical_features + categorical_features
    missing_features = [f for f in all_features if f not in raw_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    
    print("\nFeatures to be used:")
    print("Numerical features:", ', '.join(numerical_features))
    print("Categorical features:", ', '.join(categorical_features))
    
    # Preprocess data
    processed_data, transformer = data_prep.preprocess_data(raw_data)
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Processed features: {', '.join(processed_data.columns)}")

    # Convert target to numeric and split data
    print("\n2. Preparing target variable...")
    target_col = config['data_prep']['target_column']
    if target_col not in raw_data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
        
    target = convert_target_to_numeric(raw_data[target_col])
    
    print("\n3. Splitting data into train/test sets...")
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = data_prep.split_data(
        processed_data,
        target
    )
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Further split training data for validation
    validation_size = config.get('training', {}).get('validation_size', 0.2)
    if validation_size > 0:
        split_size = int(len(X_train) * (1 - validation_size))
        X_train, X_val = X_train[:split_size], X_train[split_size:]
        y_train, y_val = y_train[:split_size], y_train[split_size:]
        print(f"Validation set shape: {X_val.shape}")
    
    # Save preprocessed datasets
    print("\nSaving preprocessed datasets...")
    preprocessed_dir = experiment_dir / 'processed_data'
    preprocessed_dir.mkdir(exist_ok=True)
    
    pd.DataFrame(X_train, columns=processed_data.columns).to_csv(preprocessed_dir / 'X_train.csv', index=False)
    pd.DataFrame(X_test, columns=processed_data.columns).to_csv(preprocessed_dir / 'X_test.csv', index=False)
    pd.Series(y_train).to_csv(preprocessed_dir / 'y_train.csv', index=False)
    pd.Series(y_test).to_csv(preprocessed_dir / 'y_test.csv', index=False)
    
    if validation_size > 0:
        pd.DataFrame(X_val, columns=processed_data.columns).to_csv(preprocessed_dir / 'X_val.csv', index=False)
        pd.Series(y_val).to_csv(preprocessed_dir / 'y_val.csv', index=False)
    
    print(f"Saved preprocessed datasets to: {preprocessed_dir}")

    # Initialize and train model
    print("\n4. Training XGBoost model...")
    model_config = config.get('model', {})
    model = CustomerRetentionModel(model_config)
    
    # Train with early stopping using validation set if available
    print("Starting model training...")
    if validation_size > 0:
        print("Using validation set for early stopping...")
        model.train(X_train, y_train, X_val, y_val)
    else:
        print("Training on full training set...")
        model.train(X_train, y_train)

    # Evaluate model
    print("\n5. Evaluating model on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    # Print and save metrics
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    metrics_path = experiment_dir / 'metrics.json'
    save_metrics(metrics, metrics_path)
    print(f"\nSaved metrics to: {metrics_path}")

    # Create model directory and save model
    model_dir = experiment_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'model.pkl'
    model.save_model(model_path)
    print(f"\nSaved trained model to: {model_path}")

    # Generate model explanations
    print("\n6. Generating SHAP explanations...")
    print("Creating feature importance plot...")
    
    feature_names = processed_data.columns.tolist()
    explainer = ModelExplainer(model, feature_names)
    explainer.initialize_explainer(X_train)
    
    # Generate and save feature importance plot
    plot_path = experiment_dir / 'feature_importance.png'
    feature_importance = explainer.generate_feature_importance(
        X_test,
        plot=True,
        save_path=plot_path
    )
    
    # Sort and print feature importance
    sorted_importance = sorted(
        feature_importance.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    print("\nTop 10 Most Important Features:")
    for feature, importance in sorted_importance[:10]:
        print(f"{feature}: {importance:.4f}")
    
    print(f"\nSaved feature importance plot to: {plot_path}")
    
    # Print experiment summary
    print(f"\nExperiment completed successfully!")
    print(f"All artifacts saved to: {experiment_dir}")

    return model, explainer

def predict(config: Dict[str, Any], data_prep: DataPreparation, 
           experiment_dir: Path, as_labels: bool = False) -> None:
    """
    Load a trained model and make predictions on new data.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        data_prep (DataPreparation): Initialized data preparation pipeline
        experiment_dir (Path): Directory to save outputs
        as_labels (bool): Whether to output Yes/No instead of 1/0
    """
    # Find the latest experiment if no specific model path is given
    if 'model_path' not in config:
        latest_exp = find_latest_experiment(config.get('results_dir', 'results/'))
        if latest_exp is None:
            raise FileNotFoundError("No trained model found. Please train a model first.")
        model_path = latest_exp / 'models' / 'model.pkl'
    else:
        model_path = Path(config['model_path'])
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    print("\n1. Loading trained model...")
    model = CustomerRetentionModel(config.get('model', {}))
    model.load_model(model_path)
    print(f"Loaded model from: {model_path}")

    # Load and preprocess new data
    print("\n2. Loading and preprocessing new data...")
    raw_data = data_prep.load_data(config['data_path'])
    print(f"Loaded data shape: {raw_data.shape}")
    
    # Verify all required features are present
    feature_config = config['data_prep']['preprocessing']
    numerical_features = feature_config['numerical_features']
    categorical_features = feature_config['categorical_features']
    all_features = numerical_features + categorical_features
    
    missing_features = [f for f in all_features if f not in raw_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    
    print("\nFeatures to be used:")
    print("Numerical features:", ', '.join(numerical_features))
    print("Categorical features:", ', '.join(categorical_features))
    
    # Preprocess data
    processed_data, _ = data_prep.preprocess_data(raw_data)
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Processed features: {', '.join(processed_data.columns)}")

    # Generate predictions
    print("\n3. Generating predictions...")
    predictions = model.predict(processed_data)
    
    # Convert to labels if requested
    if as_labels:
        predictions = np.where(predictions == 1, 'Yes', 'No')
    
    # Save and display predictions
    output_path = experiment_dir / 'predictions.csv'
    pd.DataFrame(predictions, columns=['prediction']).to_csv(output_path, index=False)
    print(f"\nPrediction counts:")
    print(pd.Series(predictions).value_counts())
    print(f"\nSaved predictions to: {output_path}")

def explain_predictions(config: Dict[str, Any], data_prep: DataPreparation,
                      experiment_dir: Path) -> None:
    """
    Generate explanations for predictions on new data.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        data_prep (DataPreparation): Initialized data preparation pipeline
        experiment_dir (Path): Directory to save outputs
    """
    # Load model
    logger.info("Loading model...")
    model_path = Path(config.get('model_path'))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = CustomerRetentionModel(config.get('model', {}))
    model.load(model_path)

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    raw_data = data_prep.load_data(config['data_path'])
    processed_data, _ = data_prep.preprocess_data(raw_data)

    # Generate explanations
    logger.info("Generating explanations...")
    feature_names = processed_data.columns.tolist()
    explainer = ModelExplainer(model, feature_names)
    explainer.initialize_explainer(processed_data)
    
    for plot_type in config.get('explainability', {}).get('shap_plots', ['summary']):
        plot_path = experiment_dir / f'shap_{plot_type}.png'
        explainer.generate_feature_importance(
            processed_data,
            plot_type=plot_type,
            save_path=plot_path
        )
        logger.info(f"Saved {plot_type} plot to {plot_path}")

def main() -> None:
    """Main execution function."""
    try:
        print("\nCustomer Retention Predictor")
        print("=" * 30)
        
        # Parse arguments and load configuration
        args = parse_arguments()
        print(f"\nLoading configuration from: {args.config}")
        config = load_config(args.config)

        # Set up logging
        log_dir = Path(config.get('logs_dir', 'logs'))
        log_path = log_dir / 'pipeline.log'
        setup_logging(log_path)
        print(f"Logging to: {log_path}")

        # Create experiment directory
        results_dir = config.get('results_dir', 'results/')
        experiment_dir = setup_experiment_dir(results_dir)

        # Initialize data preparation
        print("\nInitializing data preprocessing pipeline...")
        data_prep = DataPreparation(config)

        if args.mode == 'train':
            train_model(config, data_prep, experiment_dir)
            
        elif args.mode == 'predict':
            predict(config, data_prep, experiment_dir, args.output_labels)
            
        elif args.mode == 'explain':
            explain_predictions(config, data_prep, experiment_dir)

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise

    # Initialize data preparation
    data_prep = DataPreparation(config)

    if args.mode == 'train':
        # Load and preprocess data
        raw_data = data_prep.load_data(config['data_path'])
        processed_data, transformer = data_prep.preprocess_data(raw_data)

        # Get target variable
        target_col = config['data_prep']['target_column']
        target = raw_data[target_col]

        # Convert 'Yes'/'No' to 1/0
        target = target.map({'No': 0, 'Yes': 1})

        # Split data
        X_train, X_test, y_train, y_test = data_prep.split_data(
            processed_data,
            target
        )

        # Initialize and train model
        model_config = config.get('model', {})
        model = CustomerRetentionModel(model_config)
        model.train(X_train, y_train)

        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        save_metrics(metrics, experiment_dir / 'metrics.json')

        # Generate explanations
        feature_names = processed_data.columns.tolist()
        explainer = ModelExplainer(model, feature_names)
        explainer.initialize_explainer(X_train)
        explainer.generate_feature_importance(
            X_test,
            save_path=experiment_dir / 'feature_importance.png'
        )

    elif args.mode == 'predict':
        # Load trained model
        model_config = config.get('model', {})
        model = CustomerRetentionModel(model_config)
        model.load(config.get('model_path'))

        # Load and preprocess new data
        raw_data = data_prep.load_data(config['data_path'])
        processed_data, _ = data_prep.preprocess_data(raw_data)
        predictions = model.predict(processed_data[config['data_prep']['feature_columns']])

        print("Predictions:", predictions)

    elif args.mode == 'explain':
        # Load model and explain
        model_config = config.get('model', {})
        model = CustomerRetentionModel(model_config)
        model.load(config.get('model_path'))

        explainer = ModelExplainer(model, config['data_prep']['feature_columns'])
        raw_data = data_prep.load_data(config['data_path'])
        processed_data, _ = data_prep.preprocess_data(raw_data)
        explainer.initialize_explainer(processed_data[config['data_prep']['feature_columns']])
        explainer.generate_feature_importance(
            processed_data[config['data_prep']['feature_columns']],
            save_path=experiment_dir / 'feature_importance.png'
        )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        logger.info("Process interrupted by user")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        print(f"\nConfiguration Error: {str(e)}")
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error: {str(e)}")
        logger.error("Fatal error in main script", exc_info=True)
        sys.exit(1)
    else:
        print("\nPipeline completed successfully!")
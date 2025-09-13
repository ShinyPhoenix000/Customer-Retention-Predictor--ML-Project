"""
Model explainability module using SHAP (SHapley Additive exPlanations).
Provides insights into model predictions and feature importance.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from pathlib import Path

class ModelExplainer:
    """Class for generating model explanations using SHAP."""
    
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def initialize_explainer(self, X_background: pd.DataFrame) -> None:
        """Initialize SHAP TreeExplainer for the model."""
        if hasattr(self.model, 'model'):
            # If it's our CustomerRetentionModel, use the underlying XGBoost model
            self.explainer = shap.TreeExplainer(self.model.model, X_background)
        else:
            # If it's already a raw model
            self.explainer = shap.TreeExplainer(self.model, X_background)
        
    def generate_feature_importance(self, X: pd.DataFrame,
                                    plot: bool = True,
                                    save_path: Path = None) -> Dict[str, float]:
        """
        Generate global feature importance from SHAP values.
        
        Args:
            X: DataFrame of features to explain
            plot: Whether to generate plots
            save_path: Path object pointing to directory to save plots
            
        Returns:
            Dictionary of feature importance values
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle both single output (binary classification) and multi-output cases
        if isinstance(shap_values, list):
            # For multi-class, use class 1 (positive class)
            self.shap_values = shap_values[1]
        else:
            # For binary classification
            self.shap_values = shap_values
            
        # Mean absolute SHAP values for each feature
        importance_values = np.abs(self.shap_values).mean(axis=0)
        
        # Create feature importance dictionary
        importance_dict = dict(zip(self.feature_names, importance_values))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        if plot and save_path:
            # Create directories if they don't exist
            plots_dir = save_path.parent
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate and save SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, X, 
                            feature_names=self.feature_names, 
                            show=False)
            plt.savefig(str(save_path), bbox_inches='tight', dpi=300)
            plt.close()
            
            # Generate and save feature importance bar plot
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(importance_dict)), list(importance_dict.values()))
            plt.xticks(range(len(importance_dict)), list(importance_dict.keys()), rotation=45, ha='right')
            plt.title('Feature Importance (SHAP values)')
            plt.tight_layout()
            feat_imp_path = plots_dir / 'feature_importance_bar.png'
            plt.savefig(str(feat_imp_path), bbox_inches='tight', dpi=300)
            plt.close()
            
            # Save importance values to CSV
            importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['importance'])
            importance_df.to_csv(plots_dir / 'feature_importance.csv')
        elif plot:
            # Just display the plots
            plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, X, feature_names=self.feature_names)
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(importance_dict)), list(importance_dict.values()))
            plt.xticks(range(len(importance_dict)), list(importance_dict.keys()), rotation=45, ha='right')
            plt.title('Feature Importance (SHAP values)')
            plt.tight_layout()
            plt.show()
        
        return importance_dict
        
    def explain_prediction(self, instance: pd.DataFrame,
                           plot: bool = True,
                           save_path: str = None) -> Dict[str, float]:
        """Explain a single prediction using SHAP force plot."""
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        shap_vals = self.explainer.shap_values(instance)
        contributions = dict(zip(self.feature_names, shap_vals.flatten()))
        
        if plot:
            shap.force_plot(self.explainer.expected_value, shap_vals, instance, feature_names=self.feature_names, matplotlib=True, show=True)
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
        
        return contributions
        
    def save_explanations(self, save_dir: str) -> None:
        """Save SHAP values and plots to the specified directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.shap_values is not None:
            np.save(save_path / "shap_values.npy", self.shap_values)
"""
Customer Retention Predictor - ML Package
This package contains modules for data preparation, model training,
and model explanation for customer churn prediction.
"""

__version__ = '0.1.0'
__author__ = 'ShinyPhoenix000'

# Optional: expose main modules for easier imports
from .data_prep import DataPreparation
from .model import CustomerRetentionModel
from .explain import ModelExplainer
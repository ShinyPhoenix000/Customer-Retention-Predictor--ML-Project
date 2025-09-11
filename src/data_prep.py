# src/data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataPreparation:
    def __init__(self, config):
        """
        config: dictionary from YAML, should contain a 'data_prep' section with preprocessing config
        """
        self.config = config
        self.preprocessing_config = config.get('data_prep', {}).get('preprocessing', {})
        self.random_state = config.get('random_state', 42)

    def load_data(self, file_path):
        """Load CSV data into a pandas DataFrame."""
        df = pd.read_csv(file_path)
        return df

    def preprocess_data(self, df):
        """
        Preprocess the data according to the config.
        Returns processed DataFrame and the fitted transformer.
        """
        if df is None:
            raise ValueError("Input DataFrame is None!")

        # Get feature lists from config
        numerical_features = self.preprocessing_config.get('numerical_features', [])
        categorical_features = self.preprocessing_config.get('categorical_features', [])
        
        if not numerical_features and not categorical_features:
            raise ValueError("No features specified in preprocessing config!")

        # Create transformers for numerical and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # This drops columns that aren't specified in the transformers
        )

        # Fit and transform the data
        processed_array = preprocessor.fit_transform(df)
        feature_names = (
            numerical_features +
            [f"{feat}_{val}" for feat, vals in zip(categorical_features, preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_)
             for val in vals[1:]]  # Skip first category as it's dropped
        )
        processed_data = pd.DataFrame(processed_array, columns=feature_names)
        
        return processed_data, preprocessor

    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets."""
        return train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
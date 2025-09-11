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
        
        # Validate features exist in dataframe
        all_features = numerical_features + categorical_features
        missing_features = [f for f in all_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")
        
        if not numerical_features and not categorical_features:
            raise ValueError("No features specified in preprocessing config!")

        # Validate data is not empty
        if df.empty:
            raise ValueError("Input DataFrame is empty!")
            
        # Drop rows with all missing values
        df = df.dropna(how='all')

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

    def prepare_target(self, df, target_column):
        """
        Prepare target variable by converting Yes/No to 1/0.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        # Create mapping dictionary
        target_map = {'Yes': 1, 'No': 0}
        
        # Convert target and validate
        y = df[target_column].map(target_map)
        if y.isna().any():
            invalid_values = df[target_column][~df[target_column].isin(target_map.keys())].unique()
            raise ValueError(f"Invalid values in target column: {invalid_values}")
            
        return y

    def split_data(self, X, y, test_size=0.2, val_size=None):
        """
        Split data into train, validation and test sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation. If None, only do train/test split.
            
        Returns:
            If val_size is None:
                Tuple of (X_train, X_test, y_train, y_test)
            If val_size is provided:
                Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if val_size is None:
            # Only do train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            return X_train, X_test, y_train, y_test
        else:
            # First split into train+val and test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            # Then split train into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=val_size,
                random_state=self.random_state,
                stratify=y_train_val
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
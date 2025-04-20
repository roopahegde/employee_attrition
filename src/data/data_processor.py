"""
Data processing module for employee attrition prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config.model_config import DATA_PROCESSING


class DataProcessor:
    """
    Class for handling all data processing operations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the DataProcessor with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to DATA_PROCESSING.
        """
        self.config = config or DATA_PROCESSING
        self.categorical_columns = self.config["categorical_columns"]
        self.target_column = self.config["target_column"]
        self.drop_columns = self.config["drop_columns"]
        self.test_size = self.config["test_size"]
        self.random_state = self.config["random_state"]
        self.preprocessing_pipeline = None
        
    def load_data(self, file_path):
        """
        Load the dataset from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            pandas.DataFrame: Loaded dataset.
        """
        df = pd.read_csv(file_path)
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset by handling missing values, outliers, and feature engineering.
        
        Args:
            df (pandas.DataFrame): Input dataset.
            
        Returns:
            pandas.DataFrame: Preprocessed dataset.
        """
        # Create a copy to avoid modifying the original dataframe
        df_processed = df.copy()        
        
        # Drop unnecessary columns
        if self.drop_columns:
            df_processed = df_processed.drop(columns=self.drop_columns, errors='ignore')
        
        # Handle missing values
        # For this dataset, there typically aren't missing values, but we'll add code anyway
        for col in df_processed.columns:
            if df_processed[col].isna().sum() > 0:
                if col in self.categorical_columns:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Handle outliers (capping based on IQR)
        numerical_columns = df_processed.select_dtypes(include=['int32','int64', 'float64']).columns.tolist()
        numerical_columns = [col for col in numerical_columns if col != self.target_column]
        
        for col in numerical_columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR           

            df_processed[col] = np.where(df_processed[col] < lower_bound, lower_bound, df_processed[col])
            df_processed[col] = np.where(df_processed[col] > upper_bound, upper_bound, df_processed[col])

            if(col == 'monthlyincome'):
                print("upper_bound -- ", upper_bound)
                print(df_processed['monthlyincome'].max())
        
        # Convert target column to binary (0/1)
        if self.target_column in df_processed.columns:
            if df_processed[self.target_column].dtype == 'object':
                # Assuming binary classification like 'Yes'/'No'
                df_processed[self.target_column] = df_processed[self.target_column].map(
                    lambda x: 1 if x == 'Yes' else 0
                )
        
        return df_processed
    
    def create_preprocessing_pipeline(self, df):
        """
        Create a scikit-learn preprocessing pipeline for numerical and categorical features.
        
        Args:
            df (pandas.DataFrame): Input dataset to determine column types.
            
        Returns:
            sklearn.pipeline.Pipeline: Preprocessing pipeline.
        """
        # Identify numerical and categorical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_columns = [col for col in numerical_columns if col != self.target_column]
        
        # Ensure categorical columns exist in the dataframe
        categorical_columns = [col for col in self.categorical_columns if col in df.columns]
        
        # Preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('cat', categorical_transformer, categorical_columns)
            ]
        )
        
        self.preprocessing_pipeline = preprocessor
        return preprocessor
    
    def split_data(self, df):
        """
        Split the dataset into training and testing sets.
        
        Args:
            df (pandas.DataFrame): Preprocessed dataset.
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(self, file_path):
        """
        Complete data preparation workflow.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, preprocessing_pipeline)
        """
        # Load data
        df = self.load_data(file_path)
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df_processed)
        
        # Create preprocessing pipeline
        preprocessing_pipeline = self.create_preprocessing_pipeline(df_processed)
        
        return X_train, X_test, y_train, y_test, preprocessing_pipeline
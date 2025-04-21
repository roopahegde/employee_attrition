"""
Model training module for employee attrition prediction.
"""
import os
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from employee_attrition_model.config.model_config import XGBOOST_PARAMS, MODEL_PATH


class ModelTrainer:
    """
    Class for training and saving ML models.
    """
    
    def __init__(self, config=None):
        """
        Initialize the ModelTrainer with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to XGBOOST_PARAMS.
        """
        self.config = config or XGBOOST_PARAMS
        self.model = None
        self.pipeline = None
    
    def create_model(self):
        """
        Create an XGBoost model with the specified hyperparameters.
        
        Returns:
            xgboost.XGBClassifier: The initialized model.
        """
        model = xgb.XGBClassifier(**self.config)
        self.model = model
        return model
    
    def create_pipeline(self, preprocessor):
        """
        Create a full pipeline including preprocessing and model.
        
        Args:
            preprocessor: The preprocessing ColumnTransformer.
            
        Returns:
            sklearn.pipeline.Pipeline: The complete pipeline.
        """
        if self.model is None:
            self.create_model()
            
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])
        
        return self.pipeline
    
    def train(self, X_train, y_train, preprocessor=None):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            preprocessor: Optional preprocessor to create a full pipeline.
            
        Returns:
            The trained model or pipeline.
        """
        if preprocessor is not None:
            # Create and train the full pipeline
            pipeline = self.create_pipeline(preprocessor)
            pipeline.fit(X_train, y_train)
            return pipeline
        else:
            # Train just the model
            if self.model is None:
                self.create_model()
            self.model.fit(X_train, y_train)
            return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features.
            y_test: Test target.
            
        Returns:
            dict: Dictionary with evaluation metrics.
        """
        if self.pipeline is not None:
            # If we have a pipeline, use it for prediction
            y_pred = self.pipeline.predict(X_test)
            y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        elif self.model is not None:
            # If we just have a model, use it for prediction
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError("Model has not been trained yet.")
        
        # Import here to avoid circular imports
        from employee_attrition_model.utils.evaluation import calculate_metrics
        
        return calculate_metrics(y_test, y_pred, y_pred_proba)
    
    def cross_validate(self, X, y, cv=5, scoring='roc_auc'):
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features.
            y: Target.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            scoring (str, optional): Scoring metric. Defaults to 'roc_auc'.
            
        Returns:
            dict: Cross-validation results.
        """
        if self.pipeline is not None:
            cv_results = cross_val_score(self.pipeline, X, y, cv=cv, scoring=scoring)
        elif self.model is not None:
            cv_results = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        else:
            raise ValueError("Model has not been trained yet.")
        
        return {
            'mean_score': np.mean(cv_results),
            'std_score': np.std(cv_results),
            'all_scores': cv_results
        }
    
    def save_model(self, filepath=None):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str, optional): Path to save the model. Defaults to MODEL_PATH.
            
        Returns:
            str: Path where the model was saved.
        """
        if filepath is None:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            filepath = MODEL_PATH
        
        if self.pipeline is not None:
            joblib.dump(self.pipeline, filepath)
        elif self.model is not None:
            joblib.dump(self.model, filepath)
        else:
            raise ValueError("No model to save.")
        
        return filepath
    
    def load_model(self, filepath=None):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str, optional): Path to the saved model. Defaults to MODEL_PATH.
            
        Returns:
            The loaded model.
        """
        if filepath is None:
            filepath = MODEL_PATH
            
        # Print the filepath for debugging
        print(f"Attempting to load model from: {filepath}")
        
        if os.path.exists(filepath):
            loaded_model = joblib.load(filepath)
            
            # Determine if it's a pipeline or just a model
            if isinstance(loaded_model, Pipeline):
                self.pipeline = loaded_model
                self.model = loaded_model.named_steps['model']
            else:
                self.model = loaded_model
                self.pipeline = None
                
            return loaded_model
        else:
            raise FileNotFoundError(f"No model file found at {filepath}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on.
            
        Returns:
            numpy.ndarray: Predicted classes.
        """
        # Import here to avoid circular imports
        from employee_attrition_model.utils.validation import validate_input
        import pandas as pd
        
        try:
            # Validate input data
            if isinstance(X, dict):
                # Single record as dictionary
                validated_data = validate_input(X)
                X_validated = pd.DataFrame([validated_data])
            elif isinstance(X, list) and all(isinstance(item, dict) for item in X):
                # List of dictionaries
                validated_data = validate_input(X)
                X_validated = pd.DataFrame(validated_data)
            elif isinstance(X, pd.DataFrame):
                # Already a DataFrame, validate each row
                records = X.to_dict('records')
                validated_data = validate_input(records)
                X_validated = pd.DataFrame(validated_data)
            else:
                raise ValueError("Input data must be a dictionary, list of dictionaries, or DataFrame")
            
            # Make predictions using the validated data
            if self.pipeline is not None:
                return self.pipeline.predict(X_validated)
            elif self.model is not None:
                return self.model.predict(X_validated)
            else:
                raise ValueError("Model has not been trained yet.")
                
        except Exception as e:
            raise ValueError(f"Prediction failed during data validation: {str(e)}")
    
    def predict_proba(self, X):
        """
        Get probability predictions on new data.
        
        Args:
            X: Features to predict on.
            
        Returns:
            numpy.ndarray: Predicted probabilities.
        """
        # Import here to avoid circular imports
        from employee_attrition_model.utils.validation import validate_input
        import pandas as pd
        
        try:
            # Validate input data
            if isinstance(X, dict):
                # Single record as dictionary
                validated_data = validate_input(X)
                X_validated = pd.DataFrame([validated_data])
            elif isinstance(X, list) and all(isinstance(item, dict) for item in X):
                # List of dictionaries
                validated_data = validate_input(X)
                X_validated = pd.DataFrame(validated_data)
            elif isinstance(X, pd.DataFrame):
                # Already a DataFrame, validate each row
                records = X.to_dict('records')
                validated_data = validate_input(records)
                X_validated = pd.DataFrame(validated_data)
            else:
                raise ValueError("Input data must be a dictionary, list of dictionaries, or DataFrame")
            
            # Make predictions using the validated data
            if self.pipeline is not None:
                return self.pipeline.predict_proba(X_validated)
            elif self.model is not None:
                return self.model.predict_proba(X_validated)
            else:
                raise ValueError("Model has not been trained yet.")
                
        except Exception as e:
            raise ValueError(f"Prediction probability failed during data validation: {str(e)}")
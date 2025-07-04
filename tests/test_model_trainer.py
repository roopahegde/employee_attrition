"""
Tests for the model training and prediction module.
"""
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import xgboost as xgb
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from employee_attrition_model.models.model_trainer import ModelTrainer
from employee_attrition_model.config.model_config import XGBOOST_PARAMS
from employee_attrition_model.utils.validation import EmployeeData


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 200
    
    # Create sample data matching EmployeeData schema
    data = {
        'age': np.random.randint(18, 60, n_samples),
        'businesstravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n_samples),
        'dailyrate': np.random.randint(100, 1500, n_samples),
        'department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n_samples),
        'distancefromhome': np.random.randint(1, 30, n_samples),
        'education': np.random.randint(1, 5, n_samples),
        'educationfield': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'], n_samples),
        'employeecount': np.ones(n_samples, dtype=int),
        'employeenumber': np.arange(1, n_samples + 1),
        'environmentsatisfaction': np.random.randint(1, 5, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'hourlyrate': np.random.randint(30, 100, n_samples),
        'jobinvolvement': np.random.randint(1, 5, n_samples),
        'joblevel': np.random.randint(1, 6, n_samples),
        'jobrole': np.random.choice([
            'Sales Executive', 'Research Scientist', 'Laboratory Technician',
            'Manufacturing Director', 'Healthcare Representative', 'Manager',
            'Sales Representative', 'Research Director', 'Human Resources'
        ], n_samples),
        'jobsatisfaction': np.random.randint(1, 5, n_samples),
        'maritalstatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'monthlyincome': np.random.randint(2000, 20000, n_samples),
        'monthlyrate': np.random.randint(5000, 25000, n_samples),
        'numcompaniesworked': np.random.randint(0, 8, n_samples),
        'over18': np.full(n_samples, 'Y'),
        'overtime': np.random.choice(['Yes', 'No'], n_samples),
        'percentsalaryhike': np.random.randint(0, 25, n_samples),
        'performancerating': np.random.randint(1, 5, n_samples),
        'relationshipsatisfaction': np.random.randint(1, 5, n_samples),
        'standardhours': np.full(n_samples, 80),
        'stockoptionlevel': np.random.randint(0, 4, n_samples),
        'totalworkingyears': np.random.randint(0, 40, n_samples),
        'trainingtimeslastyear': np.random.randint(0, 6, n_samples),
        'worklifebalance': np.random.randint(1, 5, n_samples),
        'yearsatcompany': np.random.randint(0, 20, n_samples),
        'yearsincurrentrole': np.random.randint(0, 15, n_samples),
        'yearssincelastpromotion': np.random.randint(0, 15, n_samples),
        'yearswithcurrmanager': np.random.randint(0, 15, n_samples)
    }
    
    X_df = pd.DataFrame(data)
    
    # Create target variable (simple rule for testing)
    y = ((X_df['monthlyincome'] > 10000) & 
         (X_df['totalworkingyears'] > 10)).astype(int)
    
    return X_df, pd.Series(y)


@pytest.fixture
def sample_preprocessor(sample_data):
    """Create a sample preprocessor for testing."""
    X_df, _ = sample_data
    
    numerical_cols = [
        'age', 'dailyrate', 'distancefromhome', 'hourlyrate', 'monthlyincome',
        'monthlyrate', 'numcompaniesworked', 'percentsalaryhike', 'totalworkingyears',
        'yearsatcompany', 'yearsincurrentrole', 'yearssincelastpromotion',
        'yearswithcurrmanager'
    ]
    
    categorical_cols = [
        'businesstravel', 'department', 'educationfield', 'gender',
        'jobrole', 'maritalstatus', 'overtime'
    ]
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor


@pytest.fixture
def trained_model(sample_data, sample_preprocessor):
    """Create and train a model for testing."""
    X_df, y = sample_data
    
    # Split into train/test
    n_train = int(0.8 * len(X_df))
    X_train, X_test = X_df.iloc[:n_train], X_df.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    
    # Train the model
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, sample_preprocessor)
    
    return trainer, X_test, y_test


def test_model_trainer_initialization():
    """Test ModelTrainer initialization."""
    trainer = ModelTrainer()
    assert trainer.config == XGBOOST_PARAMS
    assert trainer.model is None
    assert trainer.pipeline is None


def test_create_model():
    """Test creating an XGBoost model."""
    trainer = ModelTrainer()
    model = trainer.create_model()
    
    assert model is not None
    assert trainer.model is not None
    assert model is trainer.model
    assert isinstance(model, xgb.XGBClassifier)
    
    # Check that hyperparameters were set correctly
    for param, value in XGBOOST_PARAMS.items():
        assert getattr(model, param) == value


def test_create_pipeline(sample_preprocessor):
    """Test creating a pipeline with preprocessing and model."""
    trainer = ModelTrainer()
    pipeline = trainer.create_pipeline(sample_preprocessor)
    
    assert pipeline is not None
    assert trainer.pipeline is not None
    assert pipeline is trainer.pipeline
    assert pipeline.named_steps['preprocessor'] is sample_preprocessor
    assert pipeline.named_steps['model'] is trainer.model
    
    # Pipeline should have exactly these two steps
    assert list(pipeline.named_steps.keys()) == ['preprocessor', 'model']


def test_evaluate(trained_model):
    """Test model evaluation."""
    trainer, X_test, y_test = trained_model
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    
    assert metrics is not None
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    
    # All metrics should be between 0 and 1
    for metric, value in metrics.items():
        assert 0 <= value <= 1


def test_save_and_load_model(trained_model, tmp_path):
    """Test saving and loading a model."""
    trainer, X_test, y_test = trained_model
    
    # Save the model
    model_path = tmp_path / "test_model.joblib"
    saved_path = trainer.save_model(model_path)
    
    assert saved_path == model_path
    assert os.path.exists(model_path)
    
    # Predictions before loading
    original_preds = trainer.predict(X_test)
    
    # Create a new trainer and load the model
    new_trainer = ModelTrainer()
    loaded_model = new_trainer.load_model(model_path)
    
    assert loaded_model is not None
    
    # Predictions after loading should be the same
    loaded_preds = new_trainer.predict(X_test)
    assert np.array_equal(original_preds, loaded_preds)


def test_predict_with_pydantic_validation():
    """Test the predict method with Pydantic validation."""
    # Create a simple model that always predicts 0
    trainer = ModelTrainer()
    trainer.model = xgb.XGBClassifier()
    
    # Create a simple preprocessor that passes through numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['age', 'monthlyincome', 'distancefromhome'])
        ]
    )
    
    # Create a pipeline with the preprocessor and model
    trainer.pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', trainer.model)
    ])
    
    # Mock the pipeline predict method to always return 0
    # This way we don't need to actually train the model
    def mock_predict(X):
        return np.zeros(len(X))
    
    def mock_predict_proba(X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 0.8  # Probability of class 0
        probs[:, 1] = 0.2  # Probability of class 1
        return probs
    
    trainer.pipeline.predict = mock_predict
    trainer.pipeline.predict_proba = mock_predict_proba
    
    # Test with a valid employee record
    valid_employee = {
        "age": 35,
        "businesstravel": "Travel_Rarely",
        "dailyrate": 1000,
        "department": "Research & Development",
        "distancefromhome": 10,
        "education": 3,
        "educationfield": "Life Sciences",
        "employeenumber": 123,
        "environmentsatisfaction": 3,
        "gender": "Male",
        "hourlyrate": 65,
        "jobinvolvement": 3,
        "joblevel": 2,
        "jobrole": "Research Scientist",
        "jobsatisfaction": 3,
        "maritalstatus": "Married",
        "monthlyincome": 5000,
        "monthlyrate": 15000,
        "numcompaniesworked": 2,
        "overtime": "No",
        "percentsalaryhike": 15,
        "performancerating": 3,
        "relationshipsatisfaction": 3,
        "stockoptionlevel": 1,
        "totalworkingyears": 10,
        "trainingtimeslastyear": 2,
        "worklifebalance": 3,
        "yearsatcompany": 5,
        "yearsincurrentrole": 3,
        "yearssincelastpromotion": 2,
        "yearswithcurrmanager": 3
    }
    
    # Test prediction
    prediction = trainer.predict(valid_employee)
    assert prediction is not None
    assert len(prediction) == 1
    
    # Test probability prediction
    proba = trainer.predict_proba(valid_employee)
    assert proba is not None
    assert proba.shape == (1, 2)
    
    # Test with a batch of employees
    batch = [valid_employee, valid_employee.copy()]
    batch_prediction = trainer.predict(batch)
    assert batch_prediction is not None
    assert len(batch_prediction) == 2
    
    # Test with invalid data
    invalid_employee = valid_employee.copy()
    invalid_employee['businesstravel'] = 'Invalid Value'
    
    with pytest.raises(ValueError):
        trainer.predict(invalid_employee)


def test_predict_with_dataframe():
    """Test the predict method with a pandas DataFrame input."""
    # Create a simple model that always predicts 0
    trainer = ModelTrainer()
    trainer.model = xgb.XGBClassifier()
    
    # Create a simple preprocessor that passes through all features
    preprocessor = ColumnTransformer(
        transformers=[
            ('pass', 'passthrough', ['age', 'monthlyincome', 'distancefromhome'])
        ]
    )
    
    # Create a pipeline with the preprocessor and model
    trainer.pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', trainer.model)
    ])
    
    # Mock the pipeline predict method to always return 0
    def mock_predict(X):
        return np.zeros(len(X))
    
    def mock_predict_proba(X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 0.8  # Probability of class 0
        probs[:, 1] = 0.2  # Probability of class 1
        return probs
    
    trainer.pipeline.predict = mock_predict
    trainer.pipeline.predict_proba = mock_predict_proba
    
    # Create a test DataFrame
    data = {
        "age": [35, 40],
        "businesstravel": ["Travel_Rarely", "Travel_Frequently"],
        "dailyrate": [1000, 1200],
        "department": ["Research & Development", "Sales"],
        "distancefromhome": [10, 15],
        "education": [3, 4],
        "educationfield": ["Life Sciences", "Medical"],
        "employeenumber": [123, 456],
        "environmentsatisfaction": [3, 4],
        "gender": ["Male", "Female"],
        "hourlyrate": [65, 70],
        "jobinvolvement": [3, 4],
        "joblevel": [2, 3],
        "jobrole": ["Research Scientist", "Sales Executive"],
        "jobsatisfaction": [3, 4],
        "maritalstatus": ["Married", "Single"],
        "monthlyincome": [5000, 6000],
        "monthlyrate": [15000, 16000],
        "numcompaniesworked": [2, 3],
        "over18": ["Y", "Y"],
        "overtime": ["No", "Yes"],
        "percentsalaryhike": [15, 16],
        "performancerating": [3, 4],
        "relationshipsatisfaction": [3, 4],
        "standardhours": [80, 80],
        "stockoptionlevel": [1, 2],
        "totalworkingyears": [10, 15],
        "trainingtimeslastyear": [2, 3],
        "worklifebalance": [3, 4],
        "yearsatcompany": [5, 6],
        "yearsincurrentrole": [3, 4],
        "yearssincelastpromotion": [2, 3],
        "yearswithcurrmanager": [3, 4]
    }
    df = pd.DataFrame(data)
    
    # Test prediction
    prediction = trainer.predict(df)
    assert prediction is not None
    assert len(prediction) == 2
    
    # Test probability prediction
    proba = trainer.predict_proba(df)
    assert proba is not None
    assert proba.shape == (2, 2)
"""
Tests for the data processing module.
"""
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from employee_attrition_model.data.data_processor import DataProcessor
from employee_attrition_model.config.model_config import DATA_PROCESSING


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(20, 60, n_samples),
        'attrition': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'businesstravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n_samples),
        'dailyrate': np.random.randint(100, 1500, n_samples),
        'department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n_samples),
        'distancefromhome': np.random.randint(1, 30, n_samples),
        'education': np.random.randint(1, 5, n_samples),
        'educationfield': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'], n_samples),
        'employeecount': np.ones(n_samples),  # Constant value
        'employeenumber': np.arange(1, n_samples + 1),
        'environmentsatisfaction': np.random.randint(1, 5, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'hourlyrate': np.random.randint(30, 100, n_samples),
        'jobinvolvement': np.random.randint(1, 5, n_samples),
        'joblevel': np.random.randint(1, 5, n_samples),
        'jobrole': np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                     'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                     'Sales Representative', 'Research Director', 'Human Resources'], n_samples),
        'jobsatisfaction': np.random.randint(1, 5, n_samples),
        'maritalstatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'monthlyincome': np.random.randint(1000, 20000, n_samples),
        'monthlyrate': np.random.randint(5000, 25000, n_samples),
        'numcompaniesworked': np.random.randint(0, 8, n_samples),
        'over18': np.array(['Y'] * n_samples),  # Constant value
        'overtime': np.random.choice(['Yes', 'No'], n_samples),
        'percentsalaryhike': np.random.randint(10, 25, n_samples),
        'performancerating': np.random.randint(1, 5, n_samples),
        'relationshipsatisfaction': np.random.randint(1, 5, n_samples),
        'standardhours': np.array([80] * n_samples),  # Constant value
        'stockoptionlevel': np.random.randint(0, 4, n_samples),
        'totalworkingyears': np.random.randint(0, 40, n_samples),
        'trainingtimeslastyear': np.random.randint(0, 7, n_samples),
        'worklifebalance': np.random.randint(1, 5, n_samples),
        'yearsatcompany': np.random.randint(0, 20, n_samples),
        'yearsincurrentrole': np.random.randint(0, 15, n_samples),
        'yearssincelastpromotion': np.random.randint(0, 15, n_samples),
        'yearswithcurrmanager': np.random.randint(0, 15, n_samples)
    }
    
    return pd.DataFrame(data)


def test_data_processor_initialization():
    """Test DataProcessor initialization."""
    processor = DataProcessor()
    assert processor.config == DATA_PROCESSING
    assert processor.categorical_columns == DATA_PROCESSING["categorical_columns"]
    assert processor.target_column == DATA_PROCESSING["target_column"]
    assert processor.test_size == DATA_PROCESSING["test_size"]
    assert processor.random_state == DATA_PROCESSING["random_state"]


def test_load_data(tmp_path, sample_data):
    """Test loading data from a CSV file."""
    # Create a temporary CSV file
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Load the data
    processor = DataProcessor()
    loaded_data = processor.load_data(csv_path)
    
    # Check that the loaded data matches the original
    assert loaded_data.shape == sample_data.shape
    assert set(loaded_data.columns) == set(sample_data.columns)


def test_preprocess_data(sample_data):
    """Test preprocessing the data."""
    processor = DataProcessor()
    processed_data = processor.preprocess_data(sample_data)
    
    # Check that the target column was converted to binary
    assert processed_data[processor.target_column].dtype in [np.int64, np.int32, np.int8, bool]
    assert set(processed_data[processor.target_column].unique()) <= {0, 1}
    
    # Check that columns to be dropped are removed
    for col in processor.drop_columns:
        if col in sample_data.columns:
            assert col not in processed_data.columns
    
    # Check for no missing values
    assert processed_data.isnull().sum().sum() == 0


def test_create_preprocessing_pipeline(sample_data):
    """Test creating the preprocessing pipeline."""
    processor = DataProcessor()
    preprocessor = processor.create_preprocessing_pipeline(sample_data)
    
    # Check that the preprocessor is a ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Check that the preprocessor has transformers for both numerical and categorical features
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert 'num' in transformer_names
    assert 'cat' in transformer_names
    
    # Verify numerical transformer uses StandardScaler
    num_transformer = next(transformer for name, transformer, _ in preprocessor.transformers if name == 'num')
    assert any(isinstance(step[1], StandardScaler) for step in num_transformer.steps)
    
    # Verify categorical transformer uses OneHotEncoder
    cat_transformer = next(transformer for name, transformer, _ in preprocessor.transformers if name == 'cat')
    assert any(isinstance(step[1], OneHotEncoder) for step in cat_transformer.steps)
    
    # Test preprocessing transformation
    X_sample = sample_data.drop(columns=[processor.target_column])
    transformed_X = preprocessor.fit_transform(X_sample)
    
    # Check that the transformed data has the expected shape
    assert transformed_X.shape[0] == X_sample.shape[0]  # Same number of samples
    
    # The number of features should increase due to one-hot encoding
    num_categorical_cols = len([col for col in processor.categorical_columns if col in X_sample.columns])
    num_numerical_cols = len([col for col in X_sample.columns 
                             if col not in processor.categorical_columns 
                             and col not in processor.drop_columns])
    
    # Should have more features after one-hot encoding
    assert transformed_X.shape[1] >= num_numerical_cols


def test_split_data(sample_data):
    """Test splitting the data into training and testing sets."""
    processor = DataProcessor()
    
    # Preprocess data before splitting
    processed_data = processor.preprocess_data(sample_data)
    
    # Split the data
    X_train, X_test, y_train, y_test = processor.split_data(processed_data)
    
    # Check the shapes
    assert len(X_train) + len(X_test) == len(processed_data)
    assert len(y_train) + len(y_test) == len(processed_data)
    
    # Check that the test size is approximately correct
    assert abs(len(X_test) / len(processed_data) - processor.test_size) < 0.05
    
    # Check that the targets have the right shape
    assert y_train.shape == (len(X_train),)
    assert y_test.shape == (len(X_test),)
    
    # Check stratification - proportion of positive class should be similar in train and test
    train_pos_ratio = y_train.mean() if len(y_train) > 0 else 0
    test_pos_ratio = y_test.mean() if len(y_test) > 0 else 0
    assert abs(train_pos_ratio - test_pos_ratio) < 0.1  # Allow some variation due to small test set


def test_handle_outliers(sample_data):
    """Test handling of outliers in the data."""
    # Add some extreme outliers to the data
    outlier_data = sample_data.copy()
    
    # First add the extreme outlier
    outlier_data.loc[0, 'monthlyincome'] = 1000000  # Extreme outlier
    
    # Then calculate the bounds (same order as in DataProcessor)
    Q1 = outlier_data['monthlyincome'].quantile(0.25)
    Q3 = outlier_data['monthlyincome'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    processor = DataProcessor()
    processed_data = processor.preprocess_data(outlier_data)
    
    # Check that outliers were handled (capped)
    assert processed_data['monthlyincome'].max() <= upper_bound


def test_prepare_data_end_to_end(tmp_path, sample_data):
    """Test the complete data preparation workflow."""
    # Create a temporary CSV file
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Prepare the data
    processor = DataProcessor()
    X_train, X_test, y_train, y_test, preprocessor = processor.prepare_data(csv_path)
    
    # Check that all components are returned
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert preprocessor is not None
    
    # Check that the preprocessor is a ColumnTransformer
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Verify the train/test split sizes
    n_samples = len(sample_data)
    expected_test_size = int(n_samples * processor.test_size)
    expected_train_size = n_samples - expected_test_size
    
    assert len(X_train) == expected_train_size
    assert len(X_test) == expected_test_size
    assert len(y_train) == expected_train_size
    assert len(y_test) == expected_test_size
"""
Configuration file for model hyperparameters and settings.
"""

# Data processing settings
# Data processing settings
DATA_PROCESSING = {
    "test_size": 0.2,
    "random_state": 42,
    "categorical_columns": [
        "businesstravel", "department", "educationfield", 
        "gender", "jobrole", "maritalstatus", "overtime", "over18"
    ],
    "target_column": "attrition",
    "drop_columns": ["employeecount", "standardhours"]
}

# XGBoost model hyperparameters
XGBOOST_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 100,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42
}

# Paths
MODEL_PATH = "models/xgboost_attrition_model.joblib"
DATA_PATH = "data/wa_fn_usec_hr_employee_attrition_tsv.csv"

# Evaluation settings
EVALUATION = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "cv_folds": 5
}
"""
Script to train and validate the employee attrition model.

Run this script from the project root directory:
python train_and_validate.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from employee_attrition_model.config.model_config import DATA_PATH
from employee_attrition_model.data.data_processor import DataProcessor
from employee_attrition_model.models.model_trainer import ModelTrainer
from employee_attrition_model.utils.evaluation import print_metrics, plot_confusion_matrix, print_classification_report, plot_feature_importance

def train_and_validate():
    # Make sure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    print("===== Starting Model Training and Validation =====")
    
    # 1. Data Processing
    print("\n1. Processing data...")
    data_processor = DataProcessor()
    
    # Check if the data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        print(f"Looking for data file in current directory...")
        local_data_path = "data/wa_fn_usec_hr_employee_attrition_tsv.csv"
        if os.path.exists(local_data_path):
            print(f"Found data file at {local_data_path}")
            data_path = local_data_path
        else:
            raise FileNotFoundError(f"Could not find data file at {DATA_PATH} or {local_data_path}")
    else:
        data_path = DATA_PATH
    
    # Process the data
    X_train, X_test, y_train, y_test, preprocessor = data_processor.prepare_data(data_path)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Target distribution in training set: {pd.Series(y_train).value_counts().to_dict()}")
    
    # 2. Model Training
    print("\n2. Training model...")
    model_trainer = ModelTrainer()
    pipeline = model_trainer.train(X_train, y_train, preprocessor)
    
    # 3. Model Evaluation
    print("\n3. Evaluating model...")
    metrics = model_trainer.evaluate(X_test, y_test)
    print_metrics(metrics)
    
    # Detailed classification report
    y_pred = model_trainer.predict(X_test)
    print_classification_report(y_test, y_pred)
    
    # 4. Save the model
    print("\n4. Saving model...")
    model_path = os.path.join('employee_attrition_model\\trained_models', 'employee_attrition_model.joblib')
    model_trainer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # 5. Visualize results
    '''print("\n5. Generating visualizations...")
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(y_test, y_pred)
    
    # Get feature names (this might be complex due to the preprocessing)
    try:
        # For the XGBoost model, we need to extract feature names
        xgb_model = pipeline.named_steps['model']
        
        # Get feature names after preprocessing
        feature_names = []
        
        # Extract feature names from preprocessor if possible
        for name, transformer, cols in preprocessor.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(cols))
            else:
                # Fallback if get_feature_names_out is not available
                if name == 'num':
                    feature_names.extend(cols)
                else:
                    # Try to construct one-hot encoded feature names
                    for col in cols:
                        unique_values = X_train[col].unique()
                        feature_names.extend([f"{col}_{val}" for val in unique_values])
        
        # Plot feature importance
        plot_feature_importance(xgb_model, feature_names[:len(xgb_model.feature_importances_)])
    except Exception as e:
        print(f"Could not plot feature importance: {str(e)}")'''
    
    print("\n===== Model Training and Validation Complete =====")
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    print(f"Model F1 score: {metrics['f1']:.4f}")
    print(f"Model ROC AUC: {metrics['roc_auc']:.4f}")

def predict_employee_attrition():
    # Sample employee data for prediction
    sample_data = {
        "businesstravel": "Travel_Rarely",  # Options: Travel_Rarely, Travel_Frequently, Non-Travel
        "department": "Sales",  # Options: Sales, Research & Development, Human Resources
        "educationfield": "Life Sciences",  # Options: Life Sciences, Medical, Marketing, Technical Degree, Other, Human Resources
        "gender": "Female",  # Options: Female, Male
        "jobrole": "Sales Executive",  # Options: Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, Human Resources
        "maritalstatus": "Single",  # Options: Single, Married, Divorced
        "overtime": "Yes",  # Options: Yes, No
        "over18": "Y",  # Options: Y
        
        # Numeric fields
        "age": 35,
        "dailyrate": 1000,
        "distancefromhome": 10,
        "education": 3,  # 1-5 scale
        "environmentsatisfaction": 3,  # 1-4 scale
        "hourlyrate": 65,
        "jobinvolvement": 3,  # 1-4 scale
        "joblevel": 2,  # 1-5 scale
        "jobsatisfaction": 4,  # 1-4 scale
        "monthlyincome": 5000,
        "monthlyrate": 20000,
        "numcompaniesworked": 2,
        "percentsalaryhike": 15,
        "performancerating": 3,  # 1-4 scale
        "relationshipsatisfaction": 3,  # 1-4 scale
        "stockoptionlevel": 1,  # 0-3 scale
        "totalworkingyears": 10,
        "trainingtimeslastyear": 3,
        "worklifebalance": 3,  # 1-4 scale
        "yearsatcompany": 5,
        "yearsincurrentrole": 3,
        "yearssincelastpromotion": 2,
        "yearswithcurrmanager": 3
    }

    # Make prediction
    trainer = ModelTrainer()
    prediction = trainer.predict(sample_data)
    print(f"Attrition Prediction: {'Yes' if prediction[0] == 1 else 'No'}")

    # If you want probability scores
    probability = trainer.predict_proba(sample_data)
    print(f"Probability of Attrition: {probability[0][1]:.2%}")

if __name__ == "__main__":
    #train_and_validate()
    predict_employee_attrition()
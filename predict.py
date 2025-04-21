"""

"""


from employee_attrition_model.config.model_config import DATA_PATH
from employee_attrition_model.data.data_processor import DataProcessor
from employee_attrition_model.models.model_trainer import ModelTrainer
from employee_attrition_model.utils.evaluation import print_metrics, plot_confusion_matrix, print_classification_report, plot_feature_importance


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
    predict_employee_attrition()
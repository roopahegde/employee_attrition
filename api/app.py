"""
FastAPI application for employee attrition prediction service.
"""
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List


# Import from your wheel package
from employee_attrition_model.utils.validation import EmployeeData, EmployeeBatchData, validate_input
# Import model trainer from your wheel package
from employee_attrition_model.models.model_trainer import ModelTrainer

from employee_attrition_model.config.model_config import MODEL_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Employee Attrition Predictor API",
    description="API for predicting employee attrition using machine learning",
    version="0.1.0"
)

# Define prediction response models
class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response.
    """
    prediction: bool
    probability: float
    prediction_label: str

class BatchPredictionResponse(BaseModel):
    """
    Pydantic model for batch prediction responses.
    """
    predictions: List[PredictionResponse]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy"}

# Initialize the model trainer and load the model
try:
    model_trainer = ModelTrainer()
    model_trainer.load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(employee_data: EmployeeData):
    """
    Predict attrition for a single employee.
    """
    try:
        # Convert Pydantic model to dict
        data_dict = employee_data.model_dump()
        
        # Validate input data
        validated_data = validate_input(data_dict)
        
        # Make prediction
        prediction = model_trainer.predict(validated_data)[0]
        probability = model_trainer.predict_proba(validated_data)[0][1]
        
        # Create response
        result = {
            "prediction": bool(prediction),
            "probability": float(probability),
            "prediction_label": "Attrition" if prediction else "No Attrition"
        }
        
        logger.info(f"Prediction made: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_data: EmployeeBatchData):
    """
    Predict attrition for multiple employees.
    """
    try:
        # Convert Pydantic models to dicts
        data_dicts = [employee.model_dump() for employee in batch_data.employees]
        
        # Validate input data
        validated_data = validate_input(data_dicts)
        
        # Make batch predictions
        predictions = model_trainer.predict(validated_data)
        probabilities = model_trainer.predict_proba(validated_data)
        
        # Create response
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "prediction": bool(pred),
                "probability": float(probabilities[i][1]),
                "prediction_label": "Attrition" if pred else "No Attrition"
            })
        
        logger.info(f"Batch prediction made for {len(results)} employees")
        return {"predictions": results}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
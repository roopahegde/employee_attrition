# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set working directory
WORKDIR /api

# Copy requirements file
COPY api_requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy the wheel file
COPY ./dist/employee_attrition_predictor-0.1.10-py3-none-any.whl .

# Install the wheel package
RUN pip install --no-cache-dir employee_attrition_predictor-0.1.10-py3-none-any.whl

# Copy the FastAPI application
COPY app.py .

# Make port 8000 available
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
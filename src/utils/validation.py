"""
Data validation utilities using Pydantic.
"""
from typing import Optional, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import pandas as pd


class EmployeeData(BaseModel):
    """
    Pydantic model for validating employee data for attrition prediction.
    """
    # Required fields based on the dataset
    age: int = Field(..., ge=18, le=100, description="Employee age")
    businesstravel: str = Field(..., description="Business travel frequency")
    dailyrate: int = Field(..., ge=0, description="Daily rate")
    department: str = Field(..., description="Department")
    distancefromhome: int = Field(..., ge=0, description="Distance from home")
    education: int = Field(..., ge=1, le=5, description="Education level (1 to 5)")
    educationfield: str = Field(..., description="Field of education")
    employeecount: Optional[int] = Field(1, description="Employee count (constant value)")
    employeenumber: Optional[int] = Field(None, description="Employee number")
    environmentsatisfaction: int = Field(..., ge=1, le=4, description="Environment satisfaction (1 to 4)")
    gender: str = Field(..., description="Gender")
    hourlyrate: int = Field(..., ge=0, description="Hourly rate")
    jobinvolvement: int = Field(..., ge=1, le=4, description="Job involvement")
    joblevel: int = Field(..., ge=1, le=5, description="Job level (1 to 5)")
    jobrole: str = Field(..., description="Job role")
    jobsatisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction (1 to 4)")
    maritalstatus: str = Field(..., description="Marital status")
    monthlyincome: int = Field(..., ge=0, description="Monthly income")
    monthlyrate: int = Field(..., ge=0, description="Monthly rate")
    numcompaniesworked: float = Field(..., ge=0, description="Number of companies worked at")
    over18: Optional[str] = Field("Y", description="Over 18 years old (constant value)")
    overtime: str = Field(..., description="Overtime (Yes/No)")
    percentsalaryhike: int = Field(..., ge=0, le=100, description="Percent salary hike")
    performancerating: int = Field(..., ge=1, le=5, description="Performance rating (1 to 5)")
    relationshipsatisfaction: int = Field(..., ge=1, le=4, description="Relationship satisfaction (1 to 4)")
    standardhours: Optional[int] = Field(80, description="Standard hours (constant value)")
    stockoptionlevel: float = Field(..., ge=0, le=3, description="Stock option level (0 to 3)")
    totalworkingyears: float = Field(..., ge=0, description="Total working years")
    trainingtimeslastyear: float = Field(..., ge=0, description="Training times last year")
    worklifebalance: int = Field(..., ge=1, le=4, description="Work-life balance (1 to 4)")
    yearsatcompany: int = Field(..., ge=0, description="Years at the company")
    yearsincurrentrole: float = Field(..., ge=0, description="Years in current role")
    yearssincelastpromotion: float = Field(..., ge=0, description="Years since last promotion")
    yearswithcurrmanager: float = Field(..., ge=0, description="Years with current manager")
    
    @field_validator('businesstravel')
    def validate_business_travel(cls, v):
        valid_values = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']
        if v not in valid_values:
            raise ValueError(f'Business travel must be one of {valid_values}')
        return v
    
    @field_validator('department')
    def validate_department(cls, v):
        valid_values = ['Sales', 'Research & Development', 'Human Resources']
        if v not in valid_values:
            raise ValueError(f'Department must be one of {valid_values}')
        return v
    
    @field_validator('educationfield')
    def validate_education_field(cls, v):
        valid_values = ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other']
        if v not in valid_values:
            raise ValueError(f'Education field must be one of {valid_values}')
        return v
    
    @field_validator('gender')
    def validate_gender(cls, v):
        valid_values = ['Male', 'Female']
        if v not in valid_values:
            raise ValueError(f'Gender must be one of {valid_values}')
        return v
    
    @field_validator('jobrole')
    def validate_job_role(cls, v):
        valid_values = [
            'Sales Executive', 'Research Scientist', 'Laboratory Technician',
            'Manufacturing Director', 'Healthcare Representative', 'Manager',
            'Sales Representative', 'Research Director', 'Human Resources'
        ]
        if v not in valid_values:
            raise ValueError(f'Job role must be one of {valid_values}')
        return v
    
    @field_validator('maritalstatus')
    def validate_marital_status(cls, v):
        valid_values = ['Single', 'Married', 'Divorced']
        if v not in valid_values:
            raise ValueError(f'Marital status must be one of {valid_values}')
        return v
    
    @field_validator('overtime')
    def validate_overtime(cls, v):
        valid_values = ['Yes', 'No']
        if v not in valid_values:
            raise ValueError(f'Overtime must be one of {valid_values}')
        return v
    
    @field_validator('over18')
    def validate_over18(cls, v):
        if v != 'Y':
            raise ValueError(f'Over18 must be "Y"')
        return v
    
    @model_validator(mode='before')
    def set_defaults_for_constants(cls, values):
        """Set default values for fields that are constants in the dataset."""
        if 'employeecount' not in values:
            values['employeecount'] = 1
        if 'over18' not in values:
            values['over18'] = 'Y'
        if 'standardhours' not in values:
            values['standardhours'] = 80
        return values
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class EmployeeBatchData(BaseModel):
    """
    Pydantic model for validating batches of employee data.
    """
    employees: List[EmployeeData]


def validate_input(data):
    """
    Validate input data using Pydantic models.
    
    Args:
        data: Input data to validate. Can be a single employee or a batch.
        
    Returns:
        dict or list: Validated data as a dictionary or list of dictionaries.
    """
    try:
        if isinstance(data, list):
            # Validate batch data
            validated_data = EmployeeBatchData(employees=data)
            return [employee.dict() for employee in validated_data.employees]
        else:
            # Validate single employee data
            validated_data = EmployeeData(**data)
            return validated_data.dict()
    except Exception as e:
        print(f"Validation error: {str(e)}")
        # Check for missing required fields
        if isinstance(data, dict):
            missing_fields = [field for field in EmployeeData.__fields__ 
                             if field not in data 
                             and EmployeeData.__fields__[field].default is ... 
                             and field not in ['employeecount', 'over18', 'standardhours']]
            if missing_fields:
                print(f"Missing required fields: {missing_fields}")
        
        # Re-raise the exception
        raise
"""
Tests for the data validation module.
"""
import pytest
from pydantic import ValidationError
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.validation import EmployeeData, EmployeeBatchData, validate_input


def test_employee_data_validation_valid():
    """Test validation with valid employee data."""
    # Create a valid employee record
    valid_data = {
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
    
    # Validate data with Pydantic model directly
    employee = EmployeeData(**valid_data)
    
    # Check that all fields are set correctly
    assert employee.age == 35
    assert employee.businesstravel == "Travel_Rarely"
    assert employee.department == "Research & Development"
    
    # Check default values
    assert employee.over18 == "Y"
    assert employee.employeecount == 1
    assert employee.standardhours == 80


def test_employee_data_validation_defaults():
    """Test that default values are applied correctly."""
    # Create data with omitted fields that have defaults
    minimal_data = {
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
        # Note: employeecount, over18, standardhours are omitted
    }
    
    # Validate using the function
    validated_data = validate_input(minimal_data)
    
    # Check default values
    assert validated_data['over18'] == "Y"
    assert validated_data['employeecount'] == 1
    assert validated_data['standardhours'] == 80


def test_employee_data_validation_invalid_categorical():
    """Test validation with invalid categorical data."""
    # Create data with invalid categorical values
    invalid_categorical_data = {
        "age": 35,
        "businesstravel": "Invalid Travel Value",  # Invalid value
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
    
    # Validation should fail
    with pytest.raises(ValidationError):
        EmployeeData(**invalid_categorical_data)
    
    with pytest.raises(ValueError):
        validate_input(invalid_categorical_data)


def test_employee_data_validation_invalid_numerical():
    """Test validation with invalid numerical data."""
    # Create data with out-of-range numerical values
    invalid_numerical_data = {
        "age": 15,  # Too young (minimum age is 18)
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
    
    # Validation should fail
    with pytest.raises(ValidationError):
        EmployeeData(**invalid_numerical_data)
    
    with pytest.raises(ValueError):
        validate_input(invalid_numerical_data)


def test_employee_data_validation_missing_fields():
    """Test validation with missing required fields."""
    # Create data with missing required fields
    missing_fields_data = {
        "age": 35,
        "businesstravel": "Travel_Rarely",
        # "dailyrate" is missing
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
    
    # Validation should fail
    with pytest.raises(ValidationError):
        EmployeeData(**missing_fields_data)
    
    with pytest.raises(ValueError):
        validate_input(missing_fields_data)


def test_batch_validation_valid():
    """Test validation with a valid batch of employees."""
    # Create two valid employee records
    employee1 = {
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
    
    employee2 = {
        "age": 42,
        "businesstravel": "Travel_Frequently",
        "dailyrate": 1200,
        "department": "Sales",
        "distancefromhome": 5,
        "education": 4,
        "educationfield": "Medical",
        "employeenumber": 456,
        "environmentsatisfaction": 4,
        "gender": "Female",
        "hourlyrate": 70,
        "jobinvolvement": 4,
        "joblevel": 3,
        "jobrole": "Sales Executive",
        "jobsatisfaction": 4,
        "maritalstatus": "Single",
        "monthlyincome": 8000,
        "monthlyrate": 20000,
        "numcompaniesworked": 3,
        "overtime": "Yes",
        "percentsalaryhike": 20,
        "performancerating": 4,
        "relationshipsatisfaction": 4,
        "stockoptionlevel": 2,
        "totalworkingyears": 15,
        "trainingtimeslastyear": 3,
        "worklifebalance": 4,
        "yearsatcompany": 10,
        "yearsincurrentrole": 5,
        "yearssincelastpromotion": 3,
        "yearswithcurrmanager": 5
    }
    
    # Validate batch with direct model
    batch = EmployeeBatchData(employees=[employee1, employee2])
    assert len(batch.employees) == 2
    
    # Validate batch with function
    validated_batch = validate_input([employee1, employee2])
    assert len(validated_batch) == 2
    assert validated_batch[0]['age'] == 35
    assert validated_batch[1]['age'] == 42


def test_batch_validation_invalid():
    """Test batch validation with one invalid employee."""
    # Create one valid and one invalid employee
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
    
    invalid_employee = valid_employee.copy()
    invalid_employee["jobrole"] = "Invalid Job Role"  # Invalid value
    
    # Batch validation should fail with direct model
    with pytest.raises(ValidationError):
        EmployeeBatchData(employees=[valid_employee, invalid_employee])
    
    # Batch validation should fail with function
    with pytest.raises(ValueError):
        validate_input([valid_employee, invalid_employee])
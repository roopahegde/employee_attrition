#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'employee_attrition_predictor'
DESCRIPTION = "Employee attrition prediction model package "
EMAIL = "roopa.hegde@gmail.com"
AUTHOR = "Roopa Hegde"
REQUIRES_PYTHON = ">=3.9.0"


# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# Trove Classifiers: https://pypi.org/classifiers/

long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
print(f"**Root directory:** {ROOT_DIR}")
REQUIREMENTS_DIR = ROOT_DIR
PACKAGE_DIR = ROOT_DIR / 'employee_attrition_model'
about["__version__"] = "0.1.11"


# What packages are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()

# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("tests",)),
    package_data={"employee_attrition_predictor": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="BSD-3",
)
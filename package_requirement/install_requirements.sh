#!/bin/bash
python --version
pip install --upgrade azure-cli
pip install --upgrade azureml-sdk
PIP_NO_BUILD_ISOLATION=1 
pip install -r requirements_2.txt

#!/bin/bash
python --version
pip3 install --upgrade azure-cli
pip3 install --upgrade azureml-sdk
PIP_NO_BUILD_ISOLATION=1 
pip3 install -r requirements_2.txt

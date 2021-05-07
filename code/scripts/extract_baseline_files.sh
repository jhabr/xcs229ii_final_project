#!/bin/bash

azure_instance="scpdxcs@ml-lab-2c26d21d-2e78-44d8-a0be-f41571a9502f.southcentralus.cloudapp.azure.com"
project_dir="/home/scpdxcs/projects/xcs229ii_final_project"

# Extract logs from Azure Instance
scp -P 61999 -r \
  ${azure_instance}:${project_dir}/code/logs \
  ~/Downloads/xcs229ii_final_project/logs

# Extract baseline binaries (history, model weights) from Azure Instance
scp -P 61999 -r \
  ${azure_instance}:${project_dir}/code/baseline/export \
  ~/Downloads/xcs229ii_final_project/baseline/export

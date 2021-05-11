#!/bin/bash

azure_instance="scpdxcs@ml-lab-2c26d21d-2e78-44d8-a0be-f41571a9502f.southcentralus.cloudapp.azure.com"
project_dir="/home/scpdxcs/projects/xcs229ii_final_project"
export_datetime=$(date +"%Y_%m_%d_%H%M")
azure_dir=~/Downloads/xcs229ii_final_project/azure/"${export_datetime}"

# Extract logs from Azure Instance
mkdir "${azure_dir}"

scp -P 61999 -r \
  ${azure_instance}:${project_dir}/code/logs \
  "${azure_dir}"

# Extract baseline binaries (history, model weights) from Azure Instance
mkdir "${azure_dir}"

scp -P 61999 -r \
  ${azure_instance}:${project_dir}/code/experiments/export \
  "${azure_dir}"/baseline

#!/bin/bash

# Copy dataset to Azure Instance
scp -P 61999 \
  data.zip \
  scpdxcs@ml-lab-2c26d21d-2e78-44d8-a0be-f41571a9502f.southcentralus.cloudapp.azure.com:/home/scpdxcs/projects/xcs229ii_final_project/code/data

#!/bin/bash

conda --version > /dev/null
if [ $? -ne 0 ]; then
   echo "Failed to find anaconda installation. Please install anaconda"
   exit 0;
fi
# Remove VDT environment if it exists
t=$(conda env list | grep VDT)
if [ -z "$t" ]; then
   echo "No Environment found with name VDT";
else
   conda env remove -n VDT -y -q > /dev/null
fi

conda env create --file environment.yml

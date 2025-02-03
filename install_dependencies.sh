#!/bin/bash

# Update the package list
sudo yum update -y

# Install Python3 if not already installed
sudo yum install -y python3

# Upgrade pip to the latest version
sudo python3 -m pip install --upgrade pip

# Install necessary Python packages
sudo python3 -m pip install pandas matplotlib numpy

# Install PySpark
sudo python3 -m pip install pyspark

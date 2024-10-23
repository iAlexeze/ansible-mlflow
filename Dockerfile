# Use the official Python base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /mlflow

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


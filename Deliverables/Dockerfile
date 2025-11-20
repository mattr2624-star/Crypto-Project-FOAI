# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file into container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other files into the container
COPY . .

# Expose the port MLflow UI runs on
EXPOSE 5000

# Command to run MLflow UI
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]

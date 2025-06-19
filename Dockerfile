# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at the working directory
COPY requirements.txt .

# Install the packages specified in requirements.txt
# --no-cache-dir reduces image size, and --trusted-host is for network reliability
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the uvicorn server when the container launches
# It will serve your FastAPI application from frontend/main.py
CMD ["uvicorn", "frontend.main:app", "--host", "0.0.0.0", "--port", "8000"] 
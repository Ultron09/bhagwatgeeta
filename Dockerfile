# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy the current directory contents into the container at /app
COPY . /app

# Install any Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (modify if needed)
EXPOSE 8080

# Command to run the application
CMD ["python", "Main.py"]

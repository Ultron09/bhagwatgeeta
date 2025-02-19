# Use a lightweight Python image
FROM python:3.9-slim  

# Set the working directory  
WORKDIR /app  

# Install required system dependencies for OpenCV  
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*  

# Copy the project files  
COPY . /app  

# Install Python dependencies  
RUN pip install --no-cache-dir --upgrade pip  
RUN pip install --no-cache-dir -r requirements.txt  
RUN pip install --no-cache-dir opencv-python-headless  

# Expose port for Koyeb  
EXPOSE 8080  

# Run the application using Gunicorn  
CMD ["gunicorn", "-b", "0.0.0.0:5000", "Main:app"]  

# Use an official Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first (for Docker caching)
COPY requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose the Flask port
EXPOSE 5000

# Set the default command to run
CMD ["python", "src/app.py"]

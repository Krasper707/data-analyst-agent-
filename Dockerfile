# Use the official Playwright Docker image as a base for simplicity and reliability.
# It comes with all necessary system dependencies pre-installed.
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the Python packages specified in our requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# No need to run 'playwright install' as the base image already has browsers.

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run the uvicorn server when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
# Dockerfile is in configuration/
# Build context will be some_parent_directory/ (e.g., Final Work/)

FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy requirements.txt from the 'configuration' directory (relative to build context root)
# into the WORKDIR (/app) of the image.
COPY configuration/requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire 'smart-quiz' application directory (relative to build context root)
# into the WORKDIR (/app) of the image.
# This means smart-quiz/app/main.py becomes /app/app/main.py in the image.
# And smart-quiz/app/model/ becomes /app/app/model/ in the image.
# And smart-quiz/config.json becomes /app/config.json in the image.
COPY smart-quiz/ .

# Make port 8000 available to the world outside this container (documentation purpose)
EXPOSE 8000

# Define the command to run your application
# Uvicorn will look for 'app' object in 'main' module within the 'app' package.
# Since our WORKDIR is /app (which contains the contents of smart-quiz/),
# the path to the FastAPI app object is app.main:app (i.e., /app/app/main.py)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
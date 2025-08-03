FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py /app/main.py
COPY guide_HR.pdf /app/guide_HR.pdf

# Set environment variables
ENV PDF_PATH=/app/guide_HR.pdf
ENV RELOAD=false

# Expose the port
EXPOSE 8080

# Run the application using the main entry point
CMD ["python", "main.py"]
# Use a lightweight python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache Hugging Face sentence-transformer model weights for instant container startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application files
COPY . .

# ---------------------------------------------------------------
# Persistent user storage — /data survives container restarts on
# Hugging Face Docker Spaces. Create the directory and copy the
# seed users.json so it's available on first boot.
# Note: if /data is already populated from a previous run,
# the app's startup logic will NOT overwrite it.
# ---------------------------------------------------------------
RUN mkdir -p /data && cp users.json /data/users.json

# Expose the application port
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]

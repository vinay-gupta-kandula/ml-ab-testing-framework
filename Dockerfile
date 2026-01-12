FROM python:3.10-slim

# Ensure logs are visible in real-time for operational excellence
ENV PYTHONUNBUFFERED=1
# Prevent python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1. Install system dependencies if needed (e.g., for building some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip to the latest version to handle modern wheel metadata better
RUN pip install --no-cache-dir --upgrade pip

# 3. Copy requirements
COPY api/requirements.txt .

# 4. PRO-STRATEGY: Split installation to handle slow networks
# Install heavy dependencies separately to avoid massive single-point timeouts
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --default-timeout=1000 \
    --retries 10 \
    numpy pandas

RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --default-timeout=1000 \
    --retries 10 \
    scikit-learn

# 5. Install the remaining dependencies
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --default-timeout=1000 \
    --retries 10 \
    -r requirements.txt

# 6. Copy the rest of the project
COPY . .

# 7. Persistence Setup: Ensure data directory exists and is writable
# This is a critical MLOps requirement for SQLite logging
RUN mkdir -p /app/data && chmod 777 /app/data

# 8. Expose ports for both the API and the Streamlit Dashboard
EXPOSE 8000
EXPOSE 8501

# 9. Start the FastAPI application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
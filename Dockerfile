# ---------- BASE ----------
FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime deps ONLY
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install dependencies WITHOUT cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY required code
COPY noshow_iq ./noshow_iq
COPY models ./models
COPY train_model.py .
COPY pyproject.toml .

# Remove unnecessary files
RUN rm -rf /root/.cache

# Switch to non-root
USER appuser

EXPOSE 8000

CMD ["uvicorn", "noshow_iq.api:app", "--host", "0.0.0.0", "--port", "8000"]
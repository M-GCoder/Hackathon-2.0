FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Generate data and run ETL + training if not already done
RUN python data/generate_data.py && \
    python -c "from etl.pipeline import run_pipeline; run_pipeline()" && \
    python -c "from database.db_manager import DuckDBManager; db = DuckDBManager(); db.load_data(); db.close()" && \
    python mlops/train.py

# Expose ports (7860 is required by Hugging Face Spaces)
EXPOSE 8000 7860

# Create startup script
RUN echo '#!/bin/bash\n\
uvicorn api.main:app --host 0.0.0.0 --port 8000 &\n\
streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true\n' > /app/start.sh && \
    chmod +x /app/start.sh

CMD ["/bin/bash", "/app/start.sh"]

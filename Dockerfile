FROM python:3.10-slim

# Define build argument with a default empty value
ARG HF_TOKEN=""

# Set environment variable from build arg (for local builds)
# This will be overridden by RunPod's injected env vars in production
ENV HF_TOKEN=$HF_TOKEN

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir runpod transformers torch pandas scikit-learn huggingface_hub

# Copy the handler script
COPY rp_handler.py /app/


# Create a script to download the model during build
RUN echo '#!/usr/bin/env python3\n\
import os\n\
from huggingface_hub import login, snapshot_download\n\
\n\
HF_TOKEN = os.environ.get("HF_TOKEN", "")\n\
if HF_TOKEN:\n\
    login(HF_TOKEN)\n\
else:\n\
    raise EnvironmentError("HF_TOKEN is required to download the model.")\n\
\n\
MODEL_DIR = "/app/model"\n\
os.makedirs(MODEL_DIR, exist_ok=True)\n\
print("Downloading kuro-08/bert-transaction-categorization model...")\n\
snapshot_download(\n\
    repo_id="kuro-08/bert-transaction-categorization",\n\
    local_dir=MODEL_DIR,\n\
    token=HF_TOKEN,\n\
    ignore_patterns=["*.md", "*.txt", "*.pdf"]\n\
)\n\
print("Model downloaded successfully!")\n\
' > /app/download_model.py

# Make the script executable
RUN chmod +x /app/download_model.py

# Run the model download script (requires HF_TOKEN to be set in env)
RUN python /app/download_model.py

# Set the entrypoint
CMD ["python3", "-u", "/app/rp_handler.py"]
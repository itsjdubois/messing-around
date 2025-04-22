FROM python:3.10-slim

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
# Set up Hugging Face credentials if available\n\
HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN", "")\n\
if HUGGING_FACE_TOKEN:\n\
    login(HUGGING_FACE_TOKEN)\n\
\n\
MODEL_DIR = "/app/model"\n\
os.makedirs(MODEL_DIR, exist_ok=True)\n\
print("Downloading kuro-08/bert-transaction-categorization model...")\n\
snapshot_download(\n\
    repo_id="kuro-08/bert-transaction-categorization",\n\
    local_dir=MODEL_DIR,\n\
    token=HUGGING_FACE_TOKEN,\n\
    ignore_patterns=["*.md", "*.txt", "*.pdf"]\n\
)\n\
print("Model downloaded successfully!")\n\
' > /app/download_model.py

# Make the download script executable
RUN chmod +x /app/download_model.py

# Arguments that can be set when building the image
ARG HUGGING_FACE_TOKEN=""
ENV HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN}

# Run the download script during image build
RUN python /app/download_model.py

# Set the entrypoint
CMD ["python3", "-u", "/app/rp_handler.py"]
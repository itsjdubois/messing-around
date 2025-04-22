FROM python:3.10-slim AS builder

# Use ARG for build-time only
ARG HF_TOKEN=""

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir huggingface_hub

# Create the model download script
COPY download_model.py /app/

# Run the download script at build time
RUN python /app/download_model.py

# Start a fresh image for the final build
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN pip install --no-cache-dir runpod transformers torch pandas scikit-learn huggingface_hub

# Copy the handler script
COPY rp_handler.py /app/

# Copy only the downloaded model from the builder stage
COPY --from=builder /app/model /app/model

# Set the entrypoint
CMD ["python3", "-u", "/app/rp_handler.py"]
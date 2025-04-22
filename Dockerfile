FROM python:3.10-slim AS builder

ARG HF_TOKEN=""

WORKDIR /app

RUN pip install --no-cache-dir huggingface_hub

# Create the model download script
COPY download_model.py /app/

RUN python /app/download_model.py

FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir runpod transformers torch pandas scikit-learn huggingface_hub

COPY rp_handler.py /app/

COPY --from=builder /app/model /app/model

# Set the entrypoint
CMD ["python3", "-u", "/app/rp_handler.py"]
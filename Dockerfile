FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir runpod transformers torch numpy

COPY rp_handler.py /app/
COPY model/model.safetensors /app/model/

# Set the entrypoint
CMD ["python3", "-u", "/app/rp_handler.py"]
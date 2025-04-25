FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

COPY . /app/

# Pull LFS files
RUN git lfs pull

RUN pip install --no-cache-dir runpod transformers torch numpy

CMD ["python3", "-u", "/app/rp_handler.py"]
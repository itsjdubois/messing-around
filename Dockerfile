FROM python:3.10-slim

WORKDIR /app

# Install git and git-lfs
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Clone the repo and pull LFS files
RUN git clone https://github.com/itsjdubois/messing-around /app && \
    cd /app && \
    git lfs pull

# Install Python dependencies
RUN pip install --no-cache-dir runpod transformers torch numpy

# Set the entrypoint
CMD ["python3", "-u", "/app/rp_handler.py"]

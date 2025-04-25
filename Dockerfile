FROM python:3.10-slim

WORKDIR /app

# Install git and git-lfs
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Git version: $(git --version)" \
    && echo "Git LFS version: $(git lfs --version)" \
    && echo "Git location: $(which git)"

# Initialize Git LFS
RUN git lfs install

# Copy application files
COPY . /app/

# Log Git information again after copying files to ensure it's still working
RUN echo "Verifying Git after file copy: $(git --version)" \
    && echo "Git LFS status: $(git lfs install)"

# Install Python dependencies
RUN pip install --no-cache-dir runpod transformers torch numpy

# Set the entrypoint
CMD ["python3", "-u", "/app/rp_handler.py"]
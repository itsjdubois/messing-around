#!/usr/bin/env python3
import os
from huggingface_hub import login, snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(HF_TOKEN)
else:
    raise EnvironmentError("HF_TOKEN is required to download the model.")

MODEL_DIR = "/app/model"
os.makedirs(MODEL_DIR, exist_ok=True)
print("Downloading kuro-08/bert-transaction-categorization model...")
snapshot_download(
    repo_id="kuro-08/bert-transaction-categorization",
    local_dir=MODEL_DIR,
    token=HF_TOKEN,
    ignore_patterns=["*.md", "*.txt", "*.pdf"]
)
print("Model downloaded successfully!")
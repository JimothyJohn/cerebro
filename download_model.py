# download_model.py

"""
Script to download the model weights during the Docker image build process.
"""

import os
from huggingface_hub import snapshot_download

MODEL_NAME = "microsoft/Phi-3-vision-128k-instruct-onnx-cuda"
MODEL_CACHE = "model-cache"

def download_model():
    os.makedirs(MODEL_CACHE, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=MODEL_CACHE,
        local_dir=MODEL_CACHE,
        resume_download=True,
        ignore_patterns=["*.safetensors", "*.msgpack", "*.h5"]
    )

if __name__ == "__main__":
    download_model()

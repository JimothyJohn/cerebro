![cerebro](docs/cerebro.png)

# Cerebro - Dawn of Perception

[![Replicate](https://replicate.com/jimothyjohn/colmap/badge)](https://replicate.com/jimothyjohn/colmap)

This repository provides a setup for deploying Microsoft's multi-modal Phi-3 model using ONNX Runtime with CUDA acceleration. The goal is to create a Replicate A100 endpoint for efficient cloud deployment, making it easier, faster, and more accessible to utilize vision perception models for niche use cases.

## Overview

- **predict.py**: The main inference script that loads the Phi-3 model and processes predictions.
- **requirements.txt**: Lists all the dependencies required for the project.
- **cog.yaml**: Configuration file for Cog, specifying the build and run settings.

## Features

- **CUDA Acceleration**: Leverages ONNX Runtime with CUDA for faster inference on GPUs.
- **ONNX Runtime**: Uses a pre-converted ONNX model for efficient deployment.
- **Easy Deployment**: Configurations are provided for quick setup using VS Code or GitHub Codespaces.
- **Unit Testing**: Includes tests to validate the functionality of the inference script.

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- Docker installed (for devcontainer)
- VS Code with Remote - Containers extension (optional)

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/phi-3-deployment.git
   cd phi-3-deployment
   ```

2. **Using Dev Container**
    
    * Open the repository in VS Code.
    
    * When prompted, "Reopen in Container" to use the devcontainer setup.
    
    * Alternatively, you can use GitHub Codespaces for a cloud-based development environment.

3. **Install Dependencies**

    * If not using the devcontainer, create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. **Run Tests**

    * Ensure everything is set up correctly by running the unit tests:

```bash
pytest tests.py
```

5. **Running Inference**

    * You can run the predict.py script directly or set it up as part of a Cog deployment.

## Usage

The predict.py script accepts image URLs and a text prompt to generate a response using the Phi-3 model.

Example:

```py
from predict import Predictor

predictor = Predictor()
predictor.setup()
response = predictor.predict(
    image_urls="https://example.com/image1.jpg, https://example.com/image2.jpg",
    prompt="Summarize the content of these images.",
    max_new_tokens=100,
    temperature=0.5,
    do_sample=True
)

print(response)
```

### Notes

* AI-Generated Code: Most of the code in this repository is AI-generated and should be reviewed for correctness and compatibility.

* Model Files: The ONNX model files are downloaded automatically if not present in the model-cache directory.

### Contributing

Contributions are welcome! Please submit a pull request or open an issue for any changes or suggestions.

### License - MIT

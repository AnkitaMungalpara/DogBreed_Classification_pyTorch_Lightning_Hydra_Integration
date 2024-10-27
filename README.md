
# Dog Breed Classification with PyTorch Lightning and Hydra

Welcome to the **Dog Breed Classification** project! This repository highlights the journey of building a deep learning model to accurately classify dog breeds from images, integrating powerful tools like **PyTorch Lightning** for training efficiency and **Hydra** for configuration flexibility. If you're interested in replicating this setup or learning how each piece contributes, follow along as we dive into the project’s core features and setup!

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
- [Configuration with Hydra](#configuration-with-hydra)
- [Usage](#usage)
- [Testing and Code Coverage](#testing-and-code-coverage)
- [Docker Integration](#docker-integration)
- [Continuous Integration and Deployment](#continuous-integration-and-deployment)
- [References](#references)


## Project Overview

### Why Dog Breed Classification?

Dog breed classification is a popular computer vision problem that challenges a model to recognize subtle differences between breeds. This project utilizes a state-of-the-art approach to accurately classify breeds, making it a useful tool for pet identification, rescue organizations, and hobbyists alike.

### Frameworks Used

- **PyTorch Lightning**: Streamlines training and validation loops for cleaner and more manageable code.
- **Hydra**: Enables easy configuration management, helping us quickly adjust settings, model parameters, and more.
- **Docker**: Simplifies deployment by ensuring a consistent environment.
- **GitHub Actions**: Automates testing and Docker builds for continuous integration and delivery.
- **Code Coverage**: Integrated with Codecov for easy coverage monitoring.

## Directory Structure

The project is organized to separate source code, configurations, and tests, keeping everything structured for efficient development.

```plaintext
.
├── src/                # Source code for the model and training pipeline
├── tests/              # Unit and integration tests
├── configs/            # Hydra configuration files
├── Dockerfile          # Dockerfile for containerization
├── pyproject.toml      # Dependencies and project settings
├── uv.lock             # Lock file for reproducible environments with uv package manager
└── README.md           # Project documentation (you’re here!)
```

## Setup and Installation

We chose <code>uv</code>, a fast package manager, for dependency management. Here’s how to get started:

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Step 2: Install Dependencies with <code>uv</code>

[uv](https://github.com/astral-sh/uv) is a fast and lightweight Python package manager. To install dependencies, follow these steps:

```bash
uv pip install -r requirements.txt
```

For CPU-only setups, run:

```bash
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu uv pip install -r requirements.txt
```

## Configuration with Hydra

Hydra provides the flexibility to manage configurations by separating each component’s settings. You can find all configuration files in the <code>configs/</code> directory, organized as shown below:

```plaintext
configs/
├── callbacks              # Model checkpoints, early stopping, etc.
├── data                   # Dataset paths and preprocessing settings
├── eval.yaml              # Evaluation configurations
├── experiment             # Experiment-specific settings
├── hydra                  # Core Hydra settings
├── infer.yaml             # Inference parameters
├── logger                 # Logging configurations (CSV, TensorBoard)
├── model                  # Model architecture and parameters
├── paths                  # Paths for data, checkpoints, etc.
├── train.yaml             # Training parameters
└── trainer                # Trainer settings for PyTorch Lightning
```

### Example: Running with Custom Configuration

You can specify any configuration when running scripts, making it simple to experiment with different setups. For instance:

```bash
python src/train.py --config-dir configs/ --config-name train.yaml
```

## Usage

### Training the Model

To train the model, run:

```bash
python src/train.py
```

### Making Predictions

Once trained, make predictions with:

```bash
python src/predict.py --image_path path/to/your/image.jpg
```


## Testing and Code Coverage

Testing is set up with <code>pytest</code>, ensuring the model functions as expected. We also use <code>Codecov</code> to track coverage.

### Running Tests

```bash
uv run coverage run -m pytest tests/
uv run coverage report
```

### Code Coverage with Codecov

Codecov integration allows us to visualize test coverage easily and identify areas needing more testing.

## Docker Integration

This Dockerfile employs a multi-stage build process to create an efficient and lightweight image for the dog breed classification project. 

- The **build stage** utilizes a minimal Python environment with uv to install project dependencies, ensuring consistent builds through caching and synchronization. 
- The **final stage** uses a slim Python base image, installs additional libraries for evaluation, and sets up the application with appropriate environment variables. It also configures an entrypoint for running tests by default, while creating a volume for data persistence.

### Dockerfile

Here's an example of the Dockerfile setup for this project:

```dockerfile
# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
	--mount=type=bind,source=uv.lock,target=uv.lock \
	--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
	uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
ADD . /app

# Install the project and its dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
	uv sync --frozen --no-dev

# Final stage
FROM python:3.12-slim-bookworm

# Set PROJECT_ROOT environment variable
ENV PROJECT_ROOT=/app

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install additional dependencies for evaluation
RUN apt-get update && apt-get install -y \
	libgl1-mesa-glx \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

# Create a volume for data
VOLUME /app/data

# Create necessary directories
RUN mkdir -p /app/checkpoints

# Create a dummy checkpoint file for testing
RUN touch /app/checkpoints/best_model.ckpt

# Set the entrypoint to run tests
ENTRYPOINT ["pytest"]
CMD ["tests/"]
```


### Building and Running the Docker Image

To build the Docker image, run:

```bash
docker build -t <image-name> .
```

And to start the container:

```bash
docker run <image-name>
```


## Continuous Integration and Deployment

The CI/CD pipeline automates testing and deployment for the project using GitHub Actions. It triggers on pushes to the <code>main</code> branch, running tests in a clean environment and ensuring code quality through coverage reports uploaded to Codecov. Upon successful tests, it builds and pushes a Docker image to GitHub Container Registry, facilitating streamlined deployment and version management.

### GitHub Actions Workflow

Here's the <code>.github/workflows/ci.yml</code> workflow configuration:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Install uv
      uses: astral-sh/setup-uv@v2

    - name: Set up Python 3.12
      run: uv python install 3.12

    - name: Install dependencies
      env:
        UV_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu
      run: |
        uv sync --index-strategy unsafe-best-match

    - name: Run tests with coverage
      run: |
        uv run coverage run -m pytest
        uv run coverage xml -o coverage.xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        files: ./coverage.xml
        fail_ci_if_error: true

  build-and-push-image:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Log in to the Container registry
      uses: docker/login-action@65b78e6e13532edd9afa3aa52ac7964289d1a9c1
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata (tags, labels) for Docker
      id: meta
      uses: docker/metadata-action@9ec57ed1fcdbf14dcef7dfbe97b2010124a938b7
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=raw,value=session
          type=sha

    - name: Build and push Docker image
      uses: docker/build-push-action@f2a1d5e99d037542a71f64918e516c093c6f3fc4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
```

## References

- [PyTorch Lightning Documentation](https://www.pytorchlightning.ai/docs)
- [Hydra Documentation](https://hydra.cc/docs)
- [uv Package Manager](https://github.com/astral-sh/uv)
- [Dog Breed Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)

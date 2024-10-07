# Dog Breed 🐶 Classification with PyTorch Lightning and Hydra⚡

This project implements a state-of-the-art dog breed classification model using PyTorch Lightning and Hydra. It's designed to accurately identify dog breeds from images, leveraging deep learning and efficient training practices.


## 📚 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Testing](#-testing)
- [Docker](#-docker)
- [CI/CD](#-cicd)
- [Code Coverage](#-code-coverage)
- [Configuration](#️-configuration)
- [References](#-references)


## 🚀 Features

- **Advanced Dog Breed Classification**: Utilizes deep learning techniques to classify a wide range of dog breeds with high accuracy.
- **PyTorch Lightning Integration**: Employs PyTorch Lightning for streamlined and efficient model training, reducing boilerplate code and enhancing reproducibility.
- **Hydra Configuration Management**: Uses Hydra for flexible and dynamic configuration management, allowing easy experimentation with different hyperparameters and model architectures.
- **Robust CI/CD Pipeline**: Implements a comprehensive CI/CD pipeline with GitHub Actions, ensuring code quality and automated deployments.
- **Docker Containerization**: Provides a multi-stage Dockerfile for efficient and lightweight containerization.
- **Code Coverage Analysis**: Integrates with Codecov for detailed code coverage reporting, promoting high-quality, well-tested code.

## 📁 Project Structure

```
.
├── src/                # Source code for the project
├── tests/              # Unit and integration tests
├── configs/            # Hydra configuration files
├── Dockerfile          # Dockerfile for containerization
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Lock file for uv package manager
└── README.md           # This file
```

## 🛠️ Installation

This project uses `uv` as the package manager for its speed and reliability. Follow these steps to set up your environment:

1. Install `uv` if you haven't already. Visit [uv's installation guide](https://github.com/astral-sh/uv) for instructions.

2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dog-breed-classification.git
   cd dog-breed-classification
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

   For PyTorch CPU version (if you don't have a GPU), use:
   ```bash
   UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu uv pip install -r requirements.txt
   ```

## 💻 Usage

To train the model:
```bash
python src/train.py
```

To make predictions on new images:
```bash
python src/predict.py --image_path path/to/your/image.jpg
```

## 🧪 Testing

We use pytest for unit testing. To run tests with coverage:

```bash
uv run coverage run -m pytest tests/
```

To view the coverage report:
```bash
uv run coverage report
```

## 🐳 Docker

A multi-stage Dockerfile is provided to containerize the application. This ensures a lightweight and efficient container image.

To build and run the Docker image locally:

```bash
docker build -t dog-breed-classification .
docker run dog-breed-classification
```

## 🔄 CI/CD

This project uses GitHub Actions for continuous integration and deployment. The workflow includes:

- Running tests with coverage
- Building and pushing a Docker image to GitHub Container Registry (GHCR)

The CI/CD pipeline is defined in `.github/workflows/ci.yml` and includes the following steps:

1. **Test Job**:
   - Sets up the environment with Python 3.12 and uv
   - Installs dependencies
   - Runs tests with coverage
   - Uploads coverage results to Codecov

2. **Build and Push Docker Image Job**:
   - Builds the Docker image from the multi-stage Dockerfile
   - Pushes the image to GitHub Container Registry (GHCR)

## 📊 Code Coverage

Code coverage results are automatically uploaded to [Codecov](https://codecov.io) after each successful test run. This helps maintain high code quality and identifies areas that may need additional testing.

## ⚙️ Configuration

This project uses Hydra for configuration management. Configuration files can be found in the `configs/` directory. To run the project with a specific configuration:

```bash
python src/train.py --config-dir configs/ --config-name my_config
```


## 🙏 References

- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Hydra](https://hydra.cc/)
- [uv](https://github.com/astral-sh/uv)
- [Dog breed dataset](link-to-dataset) (replace with actual dataset used)




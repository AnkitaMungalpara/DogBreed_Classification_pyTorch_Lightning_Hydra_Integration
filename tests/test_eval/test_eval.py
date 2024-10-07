import pytest
import hydra
from pathlib import Path
import os
import sys
from omegaconf import OmegaConf

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Remove rootutils as it's not necessary in the Docker environment
# import rootutils
# root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import main function from eval.py
from src.eval import main

@pytest.fixture
def config():
    try:
        # Use a relative path for config_path
        with hydra.initialize(version_base=None, config_path="../../configs"):
            cfg = hydra.compose(
                config_name="eval",
                overrides=[
                    "ckpt_path=/app/checkpoints/best_model.ckpt"
                ],
            )
        return cfg
    except Exception as e:
        pytest.fail(f"Failed to initialize Hydra config: {str(e)}")

def test_eval_script(config, tmp_path):
    try:
        # Update paths for testing
        config.paths.output_dir = str(tmp_path / "output")
        config.paths.log_dir = str(tmp_path / "logs")
        config.ckpt_path = str(tmp_path / "best_model.ckpt")

        # Create a dummy checkpoint file
        Path(config.ckpt_path).touch()

        # Test different batch sizes
        for batch_size in [1, 4, 16]:
            config.batch_size = batch_size
            main(config)

        # Test with different number of workers
        for num_workers in [0, 2]:
            config.num_workers = num_workers
            main(config)

        # Verify output directory exists and contains files
        output_dir = Path(config.paths.output_dir)
        assert output_dir.exists()
        assert len(list(output_dir.glob('*'))) > 0

        # Verify log directory exists
        assert Path(config.paths.log_dir).exists()

    except Exception as e:
        pytest.fail(f"An error occurred during evaluation: {str(e)}\nConfig: {OmegaConf.to_yaml(config)}")

if __name__ == "__main__":
    pytest.main([__file__])

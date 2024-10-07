import pytest
import hydra
from pathlib import Path

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import main function from eval.py
from src.eval import main

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="eval",
            overrides=[
                "ckpt_path=/app/checkpoints/best_model.ckpt"
            ],
        )
        return cfg

def test_eval_script(config, tmp_path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")

    # Run evaluation
    main(config)

    # Add basic assertions
    assert Path(config.paths.output_dir).exists()

if __name__ == "__main__":
    pytest.main([__file__])

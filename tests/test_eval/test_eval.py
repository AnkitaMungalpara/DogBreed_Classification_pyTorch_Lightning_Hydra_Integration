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
                "ckpt_path=checkpoints/best_model.ckpt"
            ],
        )
        return cfg

def test_eval_script(config, tmp_path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")

    # Ensure the checkpoint file exists (create a dummy file for testing)
    checkpoint_path = Path(root) / "checkpoints" / "best_model.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()

    # Run evaluation
    main(config)

    # Add basic assertions
    assert Path(config.paths.output_dir).exists()

if __name__ == "__main__":
    pytest.main([__file__])

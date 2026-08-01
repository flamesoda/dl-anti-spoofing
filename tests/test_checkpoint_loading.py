import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.trainer.base_trainer import BaseTrainer


class _CheckpointOwner:
    """Minimal object accepted by BaseTrainer._from_pretrained."""

    def __init__(self, model):
        self.model = model
        self.device = "cpu"


class CheckpointLoadingTest(unittest.TestCase):
    def test_project_checkpoint_with_omegaconf_loads(self):
        source_model = torch.nn.Linear(3, 2)
        checkpoint = {
            "state_dict": source_model.state_dict(),
            "config": OmegaConf.create({"model": {"name": "test"}}),
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)

            target_model = torch.nn.Linear(3, 2)
            owner = _CheckpointOwner(target_model)
            BaseTrainer._from_pretrained(owner, checkpoint_path)

        for source, target in zip(source_model.parameters(), target_model.parameters()):
            torch.testing.assert_close(source, target)


if __name__ == "__main__":
    unittest.main()

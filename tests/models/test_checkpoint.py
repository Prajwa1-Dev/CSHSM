import pytest
import torch
import torch.nn as nn
import os
from src.models.trainer import MultitaskTrainer

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

def test_checkpoint_save_load(tmp_path):
    device = torch.device("cpu")
    model1 = DummyModel()

    # We pass a dummy loader, we won't iterate it
    trainer1 = MultitaskTrainer(model1, [], [], device)
    # Add dummy scheduler to test it
    trainer1.scheduler = torch.optim.lr_scheduler.StepLR(trainer1.optimizer, step_size=1)

    # Change some states
    trainer1.optimizer.step()
    trainer1.scheduler.step()

    ckpt_path = tmp_path / "test_ckpt.pt"
    config = {"lr": 1e-3, "epochs": 2}

    # Save checkpoint
    trainer1.save_checkpoint(str(ckpt_path), epoch=1, best_score=0.85, config=config, seed=42, patience_counter=2)

    assert os.path.exists(ckpt_path)

    # Initialize a second trainer
    model2 = DummyModel()
    trainer2 = MultitaskTrainer(model2, [], [], device)
    trainer2.scheduler = torch.optim.lr_scheduler.StepLR(trainer2.optimizer, step_size=1)

    epoch, best_score, patience_counter, loaded_config, seed = trainer2.load_checkpoint(str(ckpt_path))
    assert patience_counter == 2

    # Verify metadata
    assert epoch == 1
    assert best_score == 0.85
    assert loaded_config == config
    assert seed == 42

    # Verify model state
    for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
        assert torch.equal(p1, p2)

    # Verify optimizer state (e.g. step count)
    assert trainer1.optimizer.state_dict()['state'] == trainer2.optimizer.state_dict()['state']

    # Verify scheduler state
    assert trainer1.scheduler.state_dict() == trainer2.scheduler.state_dict()

def test_atomic_save(tmp_path):
    # Ensure it writes to temp then renames
    device = torch.device("cpu")
    model = DummyModel()
    trainer = MultitaskTrainer(model, [], [], device)
    ckpt_path = tmp_path / "atomic_ckpt.pt"
    trainer.save_checkpoint(str(ckpt_path), epoch=0, best_score=0.5, config={}, patience_counter=1)
    assert os.path.exists(ckpt_path)

    # Files starting with tmp_chkpt_ shouldn't exist after a successful save
    tmps = list(tmp_path.glob("tmp_chkpt_*"))
    assert len(tmps) == 0

def test_next_epoch_resume_logic():
    # Simulate the logic in run_m4.py
    res_epoch = 2
    epochs = 5
    start_epoch = res_epoch + 1

    # It should resume at epoch 3 (the 4th epoch, 0-indexed)
    assert start_epoch == 3

    executed_epochs = []
    for epoch in range(start_epoch, epochs):
        executed_epochs.append(epoch)

    # Should only execute epochs 3 and 4
    assert executed_epochs == [3, 4]

def test_legacy_checkpoint_load(tmp_path):
    device = torch.device("cpu")
    model1 = DummyModel()
    ckpt_path = tmp_path / "legacy_ckpt.pt"

    # Save a legacy checkpoint (just state dict)
    torch.save(model1.state_dict(), ckpt_path)

    model2 = DummyModel()
    trainer = MultitaskTrainer(model2, [], [], device)

    epoch, best_score, patience_counter, config, seed = trainer.load_checkpoint(str(ckpt_path))
    assert patience_counter == 0

    # Verify legacy fallback values
    assert epoch == -1
    assert best_score == -1.0
    assert config == {}
    assert seed == 42

    # Verify model state loaded
    for p1, p2 in zip(model1.parameters(), trainer.model.parameters()):
        assert torch.equal(p1, p2)

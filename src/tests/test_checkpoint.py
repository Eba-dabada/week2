import pytest
import torch
import torch.nn as nn
from src.train import Bank_Marketing, load_checkpoint, save_checkpoint


def test_best_model_saved(tmp_path):
    ckpt_path = tmp_path / "best_model.pt"

    x = torch.rand(8, 4)
    y = torch.randint(0, 2, (8, 1)).float()
    model_a = Bank_Marketing(num_feature=4)
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=0.001)

    loss_fn = nn.BCEWithLogitsLoss()

    model_a.train()
    optimizer_a.zero_grad()
    logits = model_a(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer_a.step()


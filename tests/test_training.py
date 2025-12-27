import pytest
import torch
from mlops_project.train import DEVICE, MyAwesomeModel

def test_device():
    device = DEVICE
    assert device == torch.device("cuda") or torch.device("cpu") or torch.device("mps")

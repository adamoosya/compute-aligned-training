"""CPU-only numerical tests; no downloads or model initialization."""

import torch


def pytest_sessionstart(session):
    torch.set_num_threads(1)

import torch


def clamp(x: torch.tensor, min_value: float = 0, max_value: float = 1):
    return torch.clamp(x, min=min_value, max=max_value)

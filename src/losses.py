import torch
from torchvision import models

MSE_LOSS = torch.nn.MSELoss()


def gram(tensor: torch.tensor) -> torch.tensor:
    B, C, H, W = tensor.shape
    x = tensor.view(C, H * W)
    return torch.mm(x, x.t())


def content_loss(g: torch.tensor, c: torch.tensor) -> torch.tensor:
    loss = MSE_LOSS(g, c)
    return loss


def style_loss(g: torch.tensor, s: torch.tensor) -> torch.tensor:
    c1, c2 = g.shape
    loss = MSE_LOSS(g, s)
    return loss / (c1**2)  # Divide by square of channels


def tv_loss(c: torch.tensor) -> torch.tensor:
    x = c[:, :, 1:, :] - c[:, :, :-1, :]
    y = c[:, :, :, 1:] - c[:, :, :, :-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss

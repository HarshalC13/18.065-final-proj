import copy
import os
from pathlib import Path
from typing import Tuple

import torch
import wandb
import wget
from fastargs import Param, Section
from fastargs.decorators import param
from torch.nn.modules.container import Sequential
from torchvision import models
from tqdm import tqdm

import utils
from losses import content_loss, gram, style_loss, tv_loss

Section("opt", "optimization arguments").params(
    opt_type=Param(str, required=True),
    lr=Param(float, required=True),
    ctn_weight=Param(float, required=True),
    sty_weight=Param(float, required=True),
    tv_weight=Param(float, required=True),
    num_iters=Param(int, required=True),
    log_freq=Param(int, required=True),
    path_to_weights=Param(str, required=True),
)


def download_vgg19() -> None:
    """
    Download Vgg19 weights (caffe weights).
    """
    proj_dir = Path(os.path.dirname(__file__)).parent  # overall project dir
    os.chdir(proj_dir)
    model_path = proj_dir.joinpath("models")
    model_path.mkdir(parents=True, exist_ok=True)
    os.chdir(model_path)
    url = "https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth"
    wget.download(url)
    os.chdir(proj_dir)


@param("opt.path_to_weights")
def load_vgg19(path_to_weights: str) -> Sequential:
    """
    Get model and load weights from path
    Args:
        path_to_weights (str): local path to the vgg19 weights.
    """
    if not Path(path_to_weights).exists():
        download_vgg19()
        raise RuntimeError(
            "Path to model weights must exist before loading! Attempting to download them now..."
        )

    vgg = models.vgg19(pretrained=False)
    vgg.load_state_dict(torch.load(path_to_weights), strict=False)
    return vgg


def get_model():
    vgg = load_vgg19()
    device = utils.check_get_device()
    vgg.features = pooling_update(vgg.features)
    model = copy.deepcopy(vgg.features)
    model.to(device)
    # no model gradients needed
    for param in model.parameters():
        param.requires_grad = False

    return model


def pooling_update(model, pool="max") -> Sequential:
    """
    Switch pooling layers to either max or avg pool layers for the given model

    Args:
        model (Sequential): Model with pooling layers
        pool (str): type of pooling to use

    Returns:
        (Sequential): model with updated pooling layers
    """
    if pool == "avg":
        ct = 0
        for layer in model.children():
            if isinstance(layer, torch.nn.MaxPool2d):
                model[ct] = torch.nn.AvgPool2d(
                    kernel_size=2, stride=2, padding=0, ceil_mode=False
                )
            ct += 1
    elif pool == "max":
        ct = 0
        for layer in model.children():
            if isinstance(layer, torch.nn.AvgPool2d):
                model[ct] = torch.nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
                )
            ct += 1

    return model


def get_features(model: Sequential, tensor: torch.tensor) -> dict:
    layers = {
        "3": "relu1_2",  # Style layers
        "8": "relu2_2",
        "17": "relu3_3",
        "26": "relu4_3",
        "35": "relu5_3",
        "22": "relu4_2",  # Content layers
        #'31' : 'relu5_2'
    }

    # Get features
    features = {}
    x = tensor
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            if name == "22":  # relu4_2
                features[layers[name]] = x
            elif name == "31":  # relu5_2
                features[layers[name]] = x
            else:
                b, c, h, w = x.shape
                features[layers[name]] = gram(x) / (h * w)

            # Terminate forward pass
            if name == "35":
                break

    return features


@param("opt.opt_type")
@param("opt.lr")
def get_optimizer(
    output_tensor: torch.tensor, opt_type: str, lr: float
) -> torch.optim.Optimizer:
    """
    Retrieves optimizer for image optimization.

    Args:
        output_tensor (torch.tensor): tensor representing image being optimized.
        opt_str (str): name of optimizer to be used.
        lr (float): learning rate for optimizer (if adam is being used).

    Returns:
        (torch.optim.Optimizer): optimizer for updating image parameters
    """
    if opt_type.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS([output_tensor])
    elif opt_type.lower() == "adam":
        optimizer = torch.optim.Adam([output_tensor], lr=lr)
    return optimizer


@param("opt.num_iters")
@param("opt.log_freq")
@param("opt.ctn_weight")
@param("opt.sty_weight")
@param("opt.tv_weight")
def stylize(
    num_iters: int,
    log_freq: int,
    output_tensor: torch.tensor,
    content_tensor: torch.tensor,
    style_tensor: torch.tensor,
    ctn_weight: float,
    sty_weight: float,
    tv_weight: float,
    model: Sequential,
    optimizer: torch.optim.Optimizer,
    titles: Tuple[str, str],
    save_path: Path,
):
    # Get features representations/Forward pass
    content_layers = ["relu4_2"]
    content_weights = {"relu4_2": 1.0}
    style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
    style_weights = {
        "relu1_2": 0.2,
        "relu2_2": 0.2,
        "relu3_3": 0.2,
        "relu4_3": 0.2,
        "relu5_3": 0.2,
    }
    c_feat = get_features(model, content_tensor)
    s_feat = get_features(model, style_tensor)

    i = [0]
    t = tqdm(range(num_iters))
    for iter in t:

        def closure():
            # Zero-out gradients
            optimizer.zero_grad()

            # Forward pass
            g_feat = get_features(model, output_tensor)

            # Compute Losses
            c_loss = 0
            s_loss = 0
            for j in content_layers:
                c_loss += content_weights[j] * content_loss(g_feat[j], c_feat[j])
            for j in style_layers:
                s_loss += style_weights[j] * style_loss(g_feat[j], s_feat[j])

            c_loss = ctn_weight * c_loss
            s_loss = sty_weight * s_loss
            t_loss = tv_weight * tv_loss(output_tensor.clone().detach())
            total_loss = c_loss + s_loss + t_loss

            # Backprop
            total_loss.backward(retain_graph=True)

            # update loss, save imgs
            if ((i[0] % log_freq) == 0) or (i[0] == num_iters):
                t.set_postfix(
                    style_loss=s_loss.item(),
                    content_loss=c_loss.item(),
                    tv_loss=t_loss,
                    total_loss=total_loss.item(),
                )
                # both color preserved and no color preserved images
                pres_clr = utils.transfer_color(
                    utils.tensor_to_img(content_tensor.clone().detach()),
                    utils.tensor_to_img(output_tensor.clone().detach()),
                )
                no_pres = utils.tensor_to_img(output_tensor.clone().detach())
                # save imgs
                og_path = Path(os.getcwd())
                os.chdir(save_path)
                utils.save_image(pres_clr, titles[0] + f"iter{i[0]}")
                utils.save_image(no_pres, titles[1] + f"iter{i[0]}")
                # wandb logs
                img_log = {
                    "color preserved": wandb.Image(
                        pres_clr, caption=f"Source Color Preserved Output Iter {i[0]}"
                    ),
                    "traditional": wandb.Image(
                        no_pres, caption=f"Traditional NS Output Iter {i[0]}"
                    ),
                }
                wandb.log(img_log)
                os.chdir(og_path)

            i[0] += 1

            return total_loss

        # Weight/Pixel update
        optimizer.step(closure)

    return output_tensor

import datetime
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from fastargs import Param, Section
from fastargs.decorators import param
from torchvision import transforms

Section("img", "image related params").params(
    max_img_size=Param(int, required=True),
    init_type=Param(str, required=True),
    pix_clip=Param(bool, required=True),
    content_path=Param(str, required=True),
    style_path=Param(str, required=True),
)


def format_time(elapsed: float):
    """
    Takes a time and formats it to hh:mm:ss:microseconds.

    Args:
        elapsed (float): time period elasped.

    Returns:
        (str): elapsed period formatted as a string.
    """
    ms = str(elapsed).split(".")[1][:6]
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded)) + ":" + ms


def check_get_device(print_device=False) -> torch.device:
    """
    Checks what device is being used.

    Args:
        print_device (bool): Whether to print the device to console or not.

    Returns:
        (torch.device): The device currently being used.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if print_device:
            print("There are %d GPU(s) available." % torch.cuda.device_count())
            print("Using the GPU:", torch.cuda.get_device_name(0))
    else:
        if print_device:
            print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return device


def load_image(path: str) -> np.ndarray:
    """
    Load image as array.

    Args:
        path (str): path to image to be loaded.

    Returns:
        (np.ndarry): image in BGR format.
    """
    # Images loaded as BGR
    img = cv2.imread(path)
    return img


@param("img.pix_clip")
def save_image(img: np.ndarray, img_title: str, pix_clip: bool) -> None:
    """
    Save image with given title in png format

    Args:
        img (np.ndarray): array representing image to be saved.
        img_title (str): title that the image will be saved under.
        pix_clip (bool): Whether to clip the array to [0, 255] bounds.
    """
    if pix_clip:
        img = img.clip(0, 255)
    cv2.imwrite(img_title + ".png", img)


@param("img.pix_clip")
def transfer_color(src: np.ndarray, dest: np.ndarray, pix_clip: bool) -> np.ndarray:
    if pix_clip:
        src, dest = src.clip(0, 255), dest.clip(0, 255)

    # Resize src to dest's size
    H, W, _ = src.shape
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_CUBIC)

    # 1 Extract the Destination's luminance
    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)
    # 2 Convert the Source from BGR to YIQ/YCbCr
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    # 3 Combine Destination's luminance and Source's IQ/CbCr
    src_yiq[..., 0] = dest_gray
    # 4 Convert new image from YIQ back to BGR
    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR)


@param("img.max_img_size")
def img_to_tensor(img: np.ndarray, max_img_size: int = 512) -> torch.tensor:
    # Rescale the image
    H, W, C = img.shape
    image_size = tuple([int((float(max_img_size) / max([H, W])) * x) for x in [H, W]])

    itot_t = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(image_size), transforms.ToTensor()]
    )
    # Subtract the means
    normalize_t = transforms.Normalize([103.939, 116.779, 123.68], [1, 1, 1])
    tensor = normalize_t(itot_t(img) * 255)
    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor


def tensor_to_img(tensor: torch.tensor) -> np.ndarray:
    # Add the means
    ttoi_t = transforms.Compose(
        [transforms.Normalize([-103.939, -116.779, -123.68], [1, 1, 1])]
    )
    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    img = ttoi_t(tensor)
    img = img.cpu().numpy()
    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img


@param("img.init_type")
def init_output(content_tensor: torch.tensor, init_type: str = "random") -> torch.tensor:
    B, C, H, W = content_tensor.shape
    if init_type == "random":
        tensor = torch.randn(C, H, W).mul(0.001).unsqueeze(0)
    else:
        tensor = content_tensor.clone().detach()

    return tensor


@param("img.content_path")
@param("img.style_path")
def get_output_title(content_path, style_path) -> Tuple[str, str]:
    """
    Generates output title

    Args:
        content_path (str): path to content image.
        style_path (str): path to style image.

    Returns:
        (str): titles for generated output image.
    """
    # drop img folder
    ctn_name = content_path.split("/")[1]
    sty_name = style_path.split("/")[1]
    # drop extension
    ctn_name = ctn_name.split(".")[0]
    sty_name = sty_name.split(".")[0]
    # combine: ctn + sty
    core_name = ctn_name + "+" + sty_name
    # make output directory
    save_path = Path(os.getcwd()).joinpath(f"outputs/{core_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    # final titles
    c_pres, no_pres = "clr-pres-" + core_name, "no-pres-" + core_name
    return c_pres, no_pres, save_path


def remove_empty_output() -> None:
    """
    Deletes all empty directories in the output folder.
    """
    for folder in os.listdir(path="outputs/"):
        try:
            os.rmdir(os.path.join(f"outputs/{folder}"))
            print("Directory '%s' has been removed successfully" % folder)
        except OSError as error:
            continue


def make_gifs(save_path) -> None:
    """
    Generates a gif from all the outputs created.

    Args:
        save_path (Path): path where outputs are being saved.
    """
    # 1. get clr_pres and no_pres separate
    pres = []
    no_pres = []
    for img in os.listdir(save_path):
        if img.startswith("clr-pres-"):
            pres.append[img]
        elif img.startswith("no-pres-"):
            no_pres.append(img)

    # 2. sort each by iteration num
    def key(s):
        if key[-4] in "0123456789":
            return (s[:-4], int(s[-4:]))
        elif key[-3] in "0123456789":
            return (s[:-3], int(s[-3:]))
        elif key[-2] in "0123456789":
            return (s[:-2], int(s[-2:]))

    pres = sorted(pres, key=key)
    no_pres = sorted(no_pres, key=key)
    # 3. generate gifs
    pres_frames = [Image.open(save_path.join(img)) for img in pres]
    no_pres_frames = [Image.open(save_path.join(img)) for img in no_pres]
    og_path = Path(os.getcwd())
    os.chdir(save_path)
    pres_one = pres_frames[0]
    pres_one.save(
        "pres-color.gif",
        format="GIF",
        append_images=pres_frames,
        save_all=True,
        duration=5000,
        loop=0,
    )
    no_one = no_pres_frames[0]
    no_one.save(
        "no-pres.gif",
        format="GIF",
        append_images=no_pres_frames,
        save_all=True,
        duration=5000,
        loop=0,
    )
    os.chdir(og_path)

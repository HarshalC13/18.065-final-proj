import datetime

import cv2
import numpy as np
import torch
from fastargs import Param, Section
from fastargs.decorators import param
from torchvision import transforms

Section("img", "image related params").params(
    max_img_size=Param(int, required=True),
    init_type=Param(str, required=True),
    pres_color=Param(bool, required=True),
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
def init_output(
    content_tensor: torch.tensor, init_type: str = "random"
) -> torch.tensor:
    B, C, H, W = content_tensor.shape
    if init_type == "random":
        tensor = torch.randn(C, H, W).mul(0.001).unsqueeze(0)
    else:
        tensor = content_tensor.clone().detach()

    return tensor

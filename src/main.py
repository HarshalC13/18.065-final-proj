from fastargs import get_current_config
from fastargs.decorators import param
import time

import utils
from model import get_model, get_optimizer, stylize


@param("img.content_path")
@param("img.style_path")
def run(content_path, style_path):
    start = time.time()
    # Load Images
    content_img = utils.load_image(content_path)
    style_img = utils.load_image(style_path)
    device = utils.check_get_device(print_device=True)
    # setup tensors
    content_tensor = utils.img_to_tensor(content_img).to(device)
    style_tensor = utils.img_to_tensor(style_img).to(device)
    g = utils.init_output(content_tensor)
    g = g.to(device).requires_grad_(True)
    # model + optimizer
    model = get_model()
    optimizer = get_optimizer(g)
    output = stylize(
        output_tensor=g,
        content_tensor=content_tensor,
        style_tensor=style_tensor,
        model=model,
        optimizer=optimizer,
    )
    print(
        "Training time was {:} (h:mm:ss:ms)".format(
            utils.format_time(time.time() - start)
        )
    )


if __name__ == "__main__":
    config = get_current_config()
    config.collect_config_file("src/config.json")
    config.validate(mode="stderr")
    config.summary()
    run()

import time
import json

import wandb
from fastargs import get_current_config
from fastargs.decorators import param

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
    *titles, save_path = utils.get_output_title()
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
        titles=titles,
        save_path=save_path,
    )
    print("Training time was {:} (h:mm:ss:ms)".format(utils.format_time(time.time() - start)))
    utils.make_gifs(save_path)
    # get rid of output folders with no outputs
    utils.remove_empty_output()


if __name__ == "__main__":
    config = get_current_config()
    config.collect_config_file("src/config.json")
    config.validate(mode="stderr")
    config.summary()
    params = json.load(open("src/config.json"))
    wandb_config = utils.update_wandb_config(params)
    wandb.init(
        project="18.065", entity="harshalc", config=wandb_config, name="dome_starry_night"
    )
    run()
    wandb.finish()

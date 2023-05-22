import os

import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from exp_runner.config import Config

EXPERIMENTS_DIR = "experiments"
CONFIG_FILE_NAME = "config.yaml"


def run(model: str, experiment: str) -> None:
    """
    Function for running experiments for the specified model
    from the directory with all information about experiment (config file)

    Parameters
    ----------
    model: str
        Model name
    experiment: str
        Directory name with the experiment config file.
        You should name this file like 'config.yaml'

    Returns
    -------
    None
    """
    config_file_path = os.path.join(
        EXPERIMENTS_DIR, model, experiment, CONFIG_FILE_NAME
    )
    config = Config.parse(config_file_path)
    exec("from " + model + ".pipeline import Pipeline")
    pipeline = eval("Pipeline(config, os.path.join(model, experiment))")
    pipeline.fit()
    pipeline.save_model(os.path.join(EXPERIMENTS_DIR, model, experiment))
    pipeline.log_hparams()


def evaluate(model: str, experiment: str, image_path: os.PathLike) -> None:
    """
    Function for evaluating for the specified model
    from the directory with all information about experiment (config file)
    for specified image

    Parameters
    ----------
    model: str
        Model name
    experiment: str
        Directory name with the experiment config file.
        You should name this file like 'config.yaml'
    image_path: os.PathLike
        Path to the image to evaluate

    Returns
    -------
    None
    """
    config_file_path = os.path.join(
        EXPERIMENTS_DIR, model, experiment, CONFIG_FILE_NAME
    )
    config = Config.parse(config_file_path)
    exec("from " + model + ".pipeline import Pipeline")
    pipeline = eval("Pipeline(config, os.path.join(model, experiment))")
    pipeline.load_model(experiment)

    image = Image.open(image_path)
    image, location = pipeline.evaluate(image)
    image, location = pipeline.inverse_transform("test", (image, location))
    fig = plt.figure(figsize=(8, 12))
    plt.imshow(image)
    plt.scatter(
        location[0][0],
        location[0][1],
        marker="+",
        s=100,
        label="Predicted",
        color="green",
    )
    plt.legend()
    plt.show()

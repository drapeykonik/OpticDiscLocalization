import os

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
    pipeline = eval("Pipeline(config)")
    # pipeline.fit()
    # pipeline.save_model(os.path.join(EXPERIMENTS_DIR, model, experiment))

    # Just example
    # image = Image.open("data/processed/localization/test/IDRiD_001.jpg")
    # tfs_image, location = pipeline.evaluate(image)
    # location = location.cpu().detach().numpy()
    # fig, axes = plt.subplots(1, 2, figsize=(12, 16))
    # axes[0].imshow(image)
    # axes[0].set_title("Original")
    # axes[1].imshow(tfs_image.permute(1, 2, 0).cpu().detach().numpy())
    # axes[1].scatter(location[0][0], location[0][1], marker="+", s=100)
    # axes[1].set_title("Transformed")
    # plt.show()

    image, location = pipeline.data_loaders["test"].dataset[0]
    src_image, src_location = pipeline.inverse_transform(
        "test", (image, location)
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 16))
    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].scatter(
        location.numpy()[0], location.numpy()[1], marker="+", s=100
    )
    axes[0].set_title("Source transformed")
    axes[1].imshow(src_image)
    axes[1].scatter(src_location[0], src_location[1], marker="+", s=100)
    axes[1].set_title("Inverse transformed")
    plt.show()

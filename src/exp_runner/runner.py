import os

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
    pipeline.fit()

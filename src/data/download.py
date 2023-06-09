"""File for dataset downloading and structure changing"""
import os
from shutil import move, rmtree
from urllib import request
from zipfile import BadZipFile, ZipFile

import progressbar


class ProgressBar:
    """
    Class for progress bar for it's usage in `urlretrieve`
    """

    def __init__(self) -> None:
        self.pbar = None

    def __call__(
        self, block_num: int, block_size: int, total_size: int
    ) -> None:
        """
        Parameters
        ----------
        block_num : int
            Blocks number of the downloaded file
        block_size : int
            Size of the one block
        total_size : int
            Total size of the downloaded file

        Returns
        -------
        None
        """
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


DOWNLOAD_ROOT = (
    "https://archive.org/download/idrid/idrid/C.%20Localization.zip"
)
DATASET_PATH = os.path.join("data", "raw")
RAW_DATASET_DIRECTORY = "C. Localization"
RAW_DATASET_LABELS_DIRECTORY = "1. Optic Disc Center Location"
RAW_DATASET_TRAIN_DIRECTORY = "a. Training Set"
RAW_DATASET_TEST_DIRECTORY = "b. Testing Set"
NEW_TRAIN_PATH = os.path.join("data", "raw", "localization", "train")
NEW_TEST_PATH = os.path.join("data", "raw", "localization", "test")


def download() -> None:
    """
    Download IDRID dataset for localization task.

    Returns
    -------
    None
    """
    os.makedirs(DATASET_PATH, exist_ok=True)
    zip_path = os.path.join(DATASET_PATH, "localization.zip")
    if not os.path.exists(zip_path):
        request.urlretrieve(DOWNLOAD_ROOT, zip_path, reporthook=ProgressBar())
    try:
        with ZipFile(zip_path) as zip_dataset:
            zip_dataset.extractall(path=os.path.join(DATASET_PATH))
            zip_dataset.close()
    except BadZipFile as error:
        print(error)
    os.makedirs(NEW_TRAIN_PATH, exist_ok=True)
    os.makedirs(NEW_TEST_PATH, exist_ok=True)
    for path, dirs, files in os.walk(
        os.path.join(DATASET_PATH, RAW_DATASET_DIRECTORY)
    ):
        if path.split(os.path.sep)[-1] == RAW_DATASET_TRAIN_DIRECTORY:
            for file in files:
                move(
                    os.path.join(path, file),
                    os.path.join(NEW_TRAIN_PATH, file),
                )
        if path.split(os.path.sep)[-1] == RAW_DATASET_TEST_DIRECTORY:
            for file in files:
                move(
                    os.path.join(path, file), os.path.join(NEW_TEST_PATH, file)
                )
        if path.split(os.path.sep)[-1] == RAW_DATASET_LABELS_DIRECTORY:
            for file in files:
                if RAW_DATASET_TRAIN_DIRECTORY[3:] in file.split("_"):
                    move(
                        os.path.join(path, file),
                        os.path.join(NEW_TRAIN_PATH, "location.csv"),
                    )
                if RAW_DATASET_TEST_DIRECTORY[3:] in file.split("_"):
                    move(
                        os.path.join(path, file),
                        os.path.join(NEW_TEST_PATH, "location.csv"),
                    )
    rmtree(os.path.join(DATASET_PATH, RAW_DATASET_DIRECTORY))


if __name__ == "__main__":
    download()

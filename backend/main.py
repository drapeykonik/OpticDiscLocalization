import os
import shutil
import sys
import uuid

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image

sys.path.append(os.getcwd() + "/")
from src.ssd.pipeline import Pipeline as SSDPipeline
from src.vggregressor.pipeline import Pipeline as VGGPipeline

CONFIGS = {
    "vgg": "models/vggregressor/config.yaml",
    "ssd": "models/ssd/config.py",
}


MODELS = {
    "vgg": VGGPipeline(CONFIGS["vgg"]),
    "ssd": SSDPipeline(CONFIGS["ssd"]),
}

app = FastAPI()


@app.get("/")
def read_root():
    return {"message:": "Welcome from the API", "cwd": os.getcwd()}


@app.post("/{model}")
def get_image(model: str, file: UploadFile = File(...)):
    MODELS[model].load_model()
    image = Image.open(file.file)
    output_image = MODELS[model].process_image(image)
    image_path = f"storage/{str(uuid.uuid4())}.jpg"
    output_image.save(image_path)
    return {"name": image_path, "size": "{0} x {1}".format(*image.size)}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8080)

import requests
import streamlit as st
from PIL import Image

MODELS = ("VGG", "SSD")

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Optical Disc Localization Web Demo")

# filetype = st.radio('Choose the file type', ('Image', 'Video', 'Camera'))

model = st.selectbox("Choose the model", [i for i in MODELS])
image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if st.button("Detect"):
    if image is not None and model is not None:
        files = {"file": image.getvalue()}
        res = requests.post(
            f"http://127.0.0.1:8080/{model.lower()}", files=files
        )
        json = res.json()

        if "message" in json:
            st.error(json.get("message"), icon="🚨")

        image = Image.open(json.get("name"))
        st.image(image)
        st.write("Size: {}".format(json.get("size")))

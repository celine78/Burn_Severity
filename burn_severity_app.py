import streamlit as st
import pandas as pd
import torch
from metrics import predict_image_pixel
from skimage import color
import skimage
from preprocessing import Preprocessing
import numpy as np
import cv2
import io
import glob
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.write("Burn Severity Assessment of Satellite Image")

uploaded_file = st.file_uploader('Satellite image upload', type=['TIFF'], accept_multiple_files=False, label_visibility="visible")

if uploaded_file is not None:
    name = uploaded_file.name
    with open(name, mode="wb") as f:
        f.write(uploaded_file.read())
    image = skimage.io.imread(name)

    model = torch.load('/Users/celine/burn-severity/model_0.pt')
    #image = skimage.io.imread('/Users/celine/Downloads/response.tiff')

    image = Preprocessing.resize_image(image)
    image = Preprocessing.delete_landsat_bands(image, [9, 12, 13])
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    mask = predict_image_pixel(model, image, device)
    mask = np.float32(mask)
    mask = color.gray2rgb(mask)
    st.image(mask, channels='RGB', caption='Burn Severity of Forest Fire')

    mask = mask * 255
    success, mask_encoded = cv2.imencode(".png", mask.astype(np.uint8))
    mask_bytes = mask_encoded.tobytes()
    st.download_button('Download the image as a PNG file', mask_bytes, file_name='burn_severity.png', mime='image/png')

import os
import cv2
import numpy as np
import torch
import skimage
from skimage import color
import streamlit as st
from torchvision import transforms
from metrics import predict_image_pixel
from preprocessing import Preprocessing


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Burn Severity Assessment")

uploaded_file = st.file_uploader("Upload a Landsat satellite image as a TIFF file with 13 bands "
                                 "[B01-B11, BQA and DataMask].", type=['TIFF'],
                                 accept_multiple_files=False,
                                 label_visibility="visible")

classes_n = st.select_slider('Select the number of burn severity levels to consider.', options=['2', '4', '5'])


def true_colors(image, contrast_value=1.4):
    true_color = np.stack([image[:, :, 3], image[:, :, 2], image[:, :, 1]], axis=-1)
    true_color = true_color / true_color.max() * contrast_value
    return true_color


if uploaded_file is not None:
    name = uploaded_file.name
    with open(name, mode="wb") as f:
        f.write(uploaded_file.read())
    image = skimage.io.imread(name)
    os.remove(name)
    if image.shape[2] != 13:
        st.warning(f'Unable to process a file with {image.shape[2]} bands.')
    else:
        if classes_n == '2':
            model = torch.load('/Users/celine/burn-severity/model_u-Net w backbone_2.pt')
        elif classes_n == '4':
            model = torch.load('/Users/celine/burn-severity/model_u-Net w backbone_4.pt')
        elif classes_n == '5':
            model = torch.load('/Users/celine/burn-severity/model_u-Net w backbone_5.pt')
        else:
            model = None
            st.warning(f'No model could be found. PLease verify that the model exists and that the path is correct.')

        image_resized = Preprocessing.resize_image(image)
        image = Preprocessing.delete_landsat_bands(image_resized, [9, 12, 13])
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        trans = transforms.Compose([
            transforms.Normalize(torch.Tensor([1166.1479, 990.1230, 860.5953, 818.1750, 2062.4404, 1656.0735,
                                               1043.4313, 830.8372, 270.5279, 269.8681]),
                                 torch.Tensor([445.0345, 427.1914, 453.4088, 571.3936, 1052.1285, 1009.8184,
                                               764.5239, 492.1391, 86.5190, 86.3010]))])
        image = trans(image)
        mask = predict_image_pixel(model, image, device)
        mask = np.float32(mask)
        mask = color.gray2rgb(mask)
        mask = ((mask - np.min(mask)) / (np.max(mask) - np.min(mask)))
        st.image(mask, channels='RGB', caption='Burn Severity of Forest Fire')

        mask_255 = mask * 255
        success, mask_encoded = cv2.imencode(".png", mask_255.astype(np.uint8))
        mask_bytes = mask_encoded.tobytes()
        st.download_button('Download the image as a PNG file', mask_bytes, file_name='burn_severity.png',
                           mime='image/png')

        image_bg = image_resized.copy()
        image_bg = np.float32(true_colors(image_bg, 1.4))

        mask_over_image = cv2.addWeighted(image_bg, 1, mask, 0.3, 0)

        st.image(mask_over_image, channels='RGB', clamp=True, caption='Burned area over satellite image')

        mask_over_image_edited = (mask_over_image * 150) + 10
        _, mask_over_image_encoded = cv2.imencode(".png", mask_over_image_edited.astype(np.uint8))
        mask_over_image_bytes = mask_over_image_encoded.tobytes()
        st.download_button('Download the image as a PNG file', mask_over_image_bytes, file_name='burn_severity.png',
                           mime='image/png')

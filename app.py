import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json

model_path = "Dog_Breed_Classification.keras"
model = tf.keras.models.load_model(model_path)

with open('label_map.json', 'r') as f:
    label_map = json.load(f)


if __name__ == "__main__":
    st.set_page_config(page_title="Dog Breed Classifier", layout='wide')

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = image[:, :, :3]
        image = np.expand_dims(image, axis=0)
        st.image(image=image, caption='Uploaded Image')

        if st.button("Classify"):
            with st.spinner("Classifying..."):
                breed = model.predict(image)
                breed = np.argmax(breed, axis=1)
                breed = label_map[str(breed[0])]
                st.success(f"Prediction: {breed}")
        else:
            st.empty()
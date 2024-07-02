import streamlit as st
import requests
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

# def predict(url, true_label):
#     response = requests.get(url)
#     response_status = response.status_code

#     if response_status == 200:
#         img = Image.open(BytesIO(response.content))
#         img = np.array(img)
#         img = np.expand_dims(img, axis=0)

#         pred_label = model.predict(img)
#         pred_label = np.argmax(pred_label, axis=1)
#         pred_label = label_map[str(pred_label[0])]

#         plt.imshow(img[0])
#         plt.title(f'True Label : {true_label} Pred Label : {pred_label}')
#         plt.axis('off')
#         plt.show()
#     else:
#         print("Failed to fetch the image.")




if __name__ == "__main__":
    # predict("https://hips.hearstapps.com/hmg-prod/images/wolf-dog-breeds-siberian-husky-1570411330.jpg?crop=1xw:0.84375xh;center,top", "Siberian Husky")
    # st.title("Dog Breed Classifier")
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
        # st.write("Classifying")
        # breed = model.predict(image)
        # breed = np.argmax(breed, axis=1)
        # breed = label_map[str(breed[0])]
        # st.markdown( """<h2 style="text-align: center;">{}</h2> """.format(breed), unsafe_allow_html=True)
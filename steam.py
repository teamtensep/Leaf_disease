import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image


# Load the trained model
model_load = tf.keras.models.load_model('saved_model')

st.title('Plant Village Images Recognizer')

# Define the class labels
labels =  ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


# Get the uploaded image file
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


if img_file_buffer is not None:
    # Open the image and convert it to a numpy array
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

# Normalization
normalization_layer = tf.keras.layers.Rescaling(1./255)

# If the "Predict" button is clicked
if st.button('Predict'):
    # view the image
    st.image(img_array)
    try:
        # Resize the image to match the input size of the model
        img_array = normalization_layer(cv2.resize(img_array.astype('uint8'), (224, 224)))

        # Add an extra dimension to represent the batch size of 1
        img_array = np.expand_dims(img_array, axis=0)

        # Get the predicted probabilities for each class
        val = model_load.predict(img_array)

        # Get the index of the class with the highest probability
        predicted_index = np.argmax(val[0])

        # Get the label corresponding to the predicted class
        predicted_label = labels[predicted_index]

        font_size = "24px"
        st.markdown("<h4 style='text-align: left; color: #2F3130; font-size: {};'>{}</h4>".format(font_size, predicted_label), unsafe_allow_html=True)
    except:
        pass

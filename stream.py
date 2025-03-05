import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model_load = tf.keras.models.load_model('model')

# Define the class labels
labels =  ['Apple_scab', 'Apple_black_rot', 'Apple_cedar_apple_rust', 'Apple_healthy', 'Corn_gray_leaf_spot', 'Corn_common_rust', 'Corn_northern_leaf_blight', 'Corn_healthy', 'Grape_black_rot', 'Grape_black_measles', 
          'Grape_leaf_blight', 'Grape_healthy', 
          'Potato_early_blight', 'Potato_healthy', 'Potato_late_blight', 'Tomato_bacterial_spot', 'Tomato_early_blight', 'Tomato_healthy', 'Tomato_late_blight', 'Tomato_leaf_mold', 'Tomato_septoria_leaf_spot',
          'Tomato_spider_mites_two-spotted_spider_mite', 'Tomato_target_spot', 'Tomato_mosaic_virus', 'Tomato_yellow_leaf_curl_virus']

# Custom CSS to style the app like a mobile phone
st.markdown(
    """
    <style>
    .mobile-frame {
        width: 375px;
        height: 667px;
        margin: auto;
        border: 16px black solid;
        border-top-width: 60px;
        border-bottom-width: 60px;
        border-radius: 36px;
        position: relative;
        background-color: white;
        padding: 20px;
        box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.5);
        overflow-y: auto;
    }
    .mobile-frame:before {
        content: '';
        display: block;
        width: 60px;
        height: 5px;
        position: absolute;
        top: -30px;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #333;
        border-radius: 10px;
    }
    .mobile-frame:after {
        content: '';
        display: block;
        width: 35px;
        height: 35px;
        position: absolute;
        left: 50%;
        bottom: -65px;
        transform: translate(-50%, -50%);
        background: #333;
        border-radius: 50%;
    }
    .button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        width: 100%;
        border-radius: 12px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
    }
    .stRadio>div {
        flex-direction: column;
        align-items: center;
    }
    .stImage>img {
        max-width: 100%;
        height: auto;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Mobile frame container
st.markdown('<div class="mobile-frame">', unsafe_allow_html=True)

# Title inside the frame
st.markdown("<h2 style='text-align: center;'>Plant Village</h2>", unsafe_allow_html=True)

# Home page options
option = st.radio("Choose an option:", ("Classify Image (Healthy/Unhealthy)", "Show Exact Class"))

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
        # View the image
        st.image(img_array, use_column_width=True)
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

            # Determine if the plant is healthy or unhealthy
            if "healthy" in predicted_label.lower():
                health_status = "Healthy"
            else:
                health_status = "Unhealthy"

            # Display the result based on the selected option
            if option == "Classify Image (Healthy/Unhealthy)":
                st.markdown(f"<h4 style='text-align: center; color: #2F3130;'>The plant is: {health_status}</h4>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h4 style='text-align: center; color: #2F3130;'>The plant is: {predicted_label}</h4>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

# Close the mobile frame container
st.markdown('</div>', unsafe_allow_html=True)

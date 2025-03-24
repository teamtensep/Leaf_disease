import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model_load = tf.keras.models.load_model('2_mnv3_best_model')

# Title of the app
st.title('Plant Village Recognizer')

# Define class labels
labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Get uploaded image
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Normalization layer
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Select mode
mode = st.radio("Choose mode:", ["Check Healthy/Unhealthy", "Predict Exact Disease"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

if st.button('Predict'):
    if img_file_buffer is not None:
        st.image(img_array, caption='Uploaded Image', use_column_width=True)
        try:
            img_array = normalization_layer(cv2.resize(img_array.astype('uint8'), (224, 224)))
            img_array = np.expand_dims(img_array, axis=0)
            val = model_load.predict(img_array)
            predicted_index = np.argmax(val[0])
            predicted_label = labels[predicted_index]
            
            if mode == "Check Healthy/Unhealthy":
                if 'healthy' in predicted_label.lower():
                    st.success("The plant is Healthy 🌿")
                else:
                    st.error("The plant is Unhealthy ❌")
            else:
                st.markdown(f"<h4 style='color: #2F3130;'>{predicted_label}</h4>", unsafe_allow_html=True)
        except:
            st.error("Error processing the image.")
    else:
        st.error("Please upload an image to proceed.")

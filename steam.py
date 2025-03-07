import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model_load = tf.keras.models.load_model('saved_model')

# Custom CSS for a realistic black mobile frame
st.markdown("""
    <style>
        .mobile-frame {
            width: 375px; /* Width of a typical mobile phone */
            height: 667px; /* Height of a typical mobile phone */
            margin: auto;
            padding: 20px;
            border-radius: 40px; /* Rounded corners for the frame */
            box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.5);
            background-color: #000000; /* Black frame */
            position: relative;
            overflow: hidden;
            border: 10px solid #333333; /* Darker border for depth */
        }
        .mobile-screen {
            width: 100%;
            height: 100%;
            background-color: #ffffff; /* White screen */
            border-radius: 30px; /* Slightly rounded corners for the screen */
            padding: 20px;
            box-sizing: border-box;
            overflow-y: auto;
            position: relative;
        }
        .mobile-screen::-webkit-scrollbar {
            display: none; /* Hide scrollbar for Chrome, Safari and Opera */
        }
        .mobile-screen {
            -ms-overflow-style: none;  /* Hide scrollbar for IE and Edge */
            scrollbar-width: none;  /* Hide scrollbar for Firefox */
        }
        .mobile-container {
            max-width: 100%;
            margin: auto;
        }
        .stButton button {
            width: 100%;
            padding: 10px;
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        /* Notch for the mobile frame */
        .notch {
            width: 60%;
            height: 20px;
            background-color: #000000;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1;
        }
        /* Speaker for the notch */
        .speaker {
            width: 50px;
            height: 5px;
            background-color: #444444;
            border-radius: 5px;
            position: absolute;
            top: 5px;
            left: 50%;
            transform: translateX(-50%);
        }
        /* Power button */
        .power-button {
            width: 5px;
            height: 40px;
            background-color: #333333;
            position: absolute;
            top: 100px;
            right: -5px;
            border-radius: 2px;
        }
        /* Volume buttons */
        .volume-buttons {
            width: 5px;
            height: 60px;
            background-color: #333333;
            position: absolute;
            top: 160px;
            right: -5px;
            border-radius: 2px;
        }
        .volume-buttons::before {
            content: '';
            width: 5px;
            height: 20px;
            background-color: #333333;
            position: absolute;
            top: -25px;
            right: 0;
            border-radius: 2px;
        }
    </style>
""", unsafe_allow_html=True)

# Mobile frame
st.markdown("""
    <div class='mobile-frame'>
        <div class='notch'>
            <div class='speaker'></div>
        </div>
        <div class='power-button'></div>
        <div class='volume-buttons'></div>
        <div class='mobile-screen'>
            <div class='mobile-container'>
""", unsafe_allow_html=True)

# Title and content inside the mobile frame
st.markdown("<h1 style='text-align: center;'>Plant Village Recognizer</h1>", unsafe_allow_html=True)

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

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

# Normalization layer
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Select mode
mode = st.radio("Choose mode:", ["Check Healthy/Unhealthy", "Predict Exact Disease"])

if st.button('Predict'):
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

# Close mobile frame
st.markdown("""
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

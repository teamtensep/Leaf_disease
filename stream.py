import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model_load = tf.keras.models.load_model('model')

# Define class labels
labels =  ['Apple_scab', 'Apple_black_rot', 'Apple_cedar_apple_rust', 'Apple_healthy', 'Corn_gray_leaf_spot', 
           'Corn_common_rust', 'Corn_northern_leaf_blight', 'Corn_healthy', 'Grape_black_rot', 'Grape_black_measles', 
           'Grape_leaf_blight', 'Grape_healthy', 'Potato_early_blight', 'Potato_healthy', 'Potato_late_blight', 
           'Tomato_bacterial_spot', 'Tomato_early_blight', 'Tomato_healthy', 'Tomato_late_blight', 'Tomato_leaf_mold', 
           'Tomato_septoria_leaf_spot', 'Tomato_spider_mites_two-spotted_spider_mite', 'Tomato_target_spot', 
           'Tomato_mosaic_virus', 'Tomato_yellow_leaf_curl_virus']

# Define Healthy/Unhealthy Categories
healthy_labels = ['Apple_healthy', 'Corn_healthy', 'Grape_healthy', 'Potato_healthy', 'Tomato_healthy']

# UI Customization for a Mobile Look
st.set_page_config(page_title="Plant Recognizer", layout="centered")

# Mobile Phone Frame Styling
st.markdown(
    """
    <style>
    .mobile-container {
        width: 375px; /* iPhone standard width */
        height: auto;
        margin: auto;
        padding: 20px;
        background: white;
        border-radius: 30px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        border: 10px solid #222; /* Black border to mimic a phone */
        position: relative;
        text-align: center;
    }

    /* Fake notch for the mobile look */
    .notch {
        width: 180px;
        height: 25px;
        background: black;
        border-radius: 10px;
        margin: auto;
        position: absolute;
        top: -12px;
        left: 50%;
        transform: translateX(-50%);
    }

    .stButton>button {
        width: 100%;
        font-size: 18px;
        padding: 12px;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
    }

    .stTitle {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }

    .stRadio {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Start the mobile frame
st.markdown('<div class="mobile-container">', unsafe_allow_html=True)
st.markdown('<div class="notch"></div>', unsafe_allow_html=True)  # Add the notch

# Home Screen with Two Options
st.title("🌱 Plant Disease Detector")

choice = st.radio("Choose an Option", ["📌 Healthy or Unhealthy", "🔍 Full Classification"], index=None)

# Image Upload Section
img_file_buffer = st.file_uploader("📸 Upload an Image", type=["png", "jpg", "jpeg"])

# If an image is uploaded, show it
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    st.image(img_array, caption="Uploaded Image", use_column_width=True)

# Function to preprocess the image
def preprocess_image(img):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    img_resized = cv2.resize(np.array(img), (224, 224))
    img_normalized = normalization_layer(img_resized)
    return np.expand_dims(img_normalized, axis=0)

# Prediction Logic
if st.button("🔍 Predict") and img_file_buffer is not None:
    try:
        processed_img = preprocess_image(image)
        val = model_load.predict(processed_img)
        predicted_index = np.argmax(val[0])
        predicted_label = labels[predicted_index]

        # Display Prediction Based on Choice
        if choice == "📌 Healthy or Unhealthy":
            health_status = "Healthy 🌿" if predicted_label in healthy_labels else "Unhealthy 🍂"
            st.success(f"**Plant Condition:** {health_status}")

        elif choice == "🔍 Full Classification":
            st.info(f"**Predicted Class:** {predicted_label}")

    except Exception as e:
        st.error("⚠️ Error in prediction. Please try again.")
        st.write(e)

# End the mobile frame
st.markdown('</div>', unsafe_allow_html=True)

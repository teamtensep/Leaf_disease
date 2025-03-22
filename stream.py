import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications import mobilenet_v3

# Load the trained TensorFlow model (not .h5)
MODEL_PATH = "new_again_final_mobilenetv3_model"
model_load = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
labels =  ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Image size for model input
IMG_SIZE = (224, 224)

# Function to preprocess an image
def preprocess_image(image):
    image = tf.image.resize(image, IMG_SIZE)
    image = mobilenet_v3.preprocess_input(image)  # MobileNetV3 preprocessing
    return image

# Streamlit UI
st.title("🌿 Plant Village - Disease Classification")

# Upload image
img_file_buffer = st.file_uploader("📸 Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    # Display uploaded image
    st.image(img_array, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        try:
            # Convert image to Tensor
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

            # Preprocess the image
            img_tensor = preprocess_image(img_tensor)

            # Expand dimensions for batch
            img_tensor = tf.expand_dims(img_tensor, axis=0)

            # Make prediction
            predictions = model_load.predict(img_tensor)
            predicted_index = np.argmax(predictions[0])
            predicted_label = labels[predicted_index]

            # Determine health status
            health_status = "✅ Healthy" if "healthy" in predicted_label.lower() else "⚠️ Unhealthy"

            # Display result
            st.markdown(f"### 🌱 **Plant Health Status:** {health_status}")
            st.markdown(f"### 🔬 **Disease Classification:** {predicted_label}")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")


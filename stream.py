import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications import mobilenet_v3

# Load the trained model
model_load = tf.keras.models.load_model("new_again_final_mobilenetv3_model")
# Define the class labels
labels =  ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

# Function to preprocess an image
def preprocess_image(image, augment=False):
    image = tf.image.resize(image, IMG_SIZE)
    
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_crop(image, size=[IMG_SIZE[0]-20, IMG_SIZE[1]-20, 3])
        image = tf.image.resize(image, IMG_SIZE)
    
    # Apply MobileNetV3 preprocessing
    image = mobilenet_v3.preprocess_input(image)
    return image

# Streamlit UI
st.title("Plant Village - Disease Classification")

# Upload image
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    # Display image
    st.image(img_array, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            # Convert to Tensor
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            
            # Preprocess the image (no augmentation for inference)
            img_tensor = preprocess_image(img_tensor, augment=False)
            
            # Expand dimensions for batch
            img_tensor = tf.expand_dims(img_tensor, axis=0)

            # Make prediction
            predictions = model_load.predict(img_tensor)
            predicted_index = np.argmax(predictions[0])
            predicted_label = labels[predicted_index]

            # Classify as Healthy or Unhealthy
            health_status = "Healthy" if "healthy" in predicted_label.lower() else "Unhealthy"

            # Display result
            st.markdown(f"**Plant Health Status:** {health_status}")
            st.markdown(f"**Exact Classification:** {predicted_label}")
        except Exception as e:
            st.error(f"Error processing image: {e}")

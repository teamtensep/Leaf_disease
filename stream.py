import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('3_new_again_final_mobilenetv3_model')

model = load_model()

# Title of the app
st.title('🌱 Plant Village Disease Detector')

# Define class labels
labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Upload image
img_file_buffer = st.file_uploader("📷 Upload an image", type=["png", "jpg", "jpeg"])

# Select mode
mode = st.radio("🎯 Choose mode:", ["Check Healthy/Unhealthy", "Predict Exact Disease"])

def preprocess_image(image):
    img_array = np.array(image)
    img_array = cv2.resize(img_array.astype('uint8'), (224, 224))
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_gradcam_heatmap(img_array, model, layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_array, heatmap):
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_array.astype('uint8'), 0.6, heatmap, 0.4, 0)
    return superimposed_img

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = preprocess_image(image)

    if st.button('🔍 Predict'):
        st.image(image, caption='Uploaded Image', use_column_width=True)

        try:
            # Make prediction
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_label = labels[predicted_index]
            confidence = predictions[0][predicted_index] * 100

            # Show top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            st.write("📊 **Top 3 Predictions:**")
            for idx in top_3_indices:
                st.write(f"🔹 {labels[idx]}: **{predictions[0][idx]*100:.2f}%**")

            # Display result
            if mode == "Check Healthy/Unhealthy":
                if 'healthy' in predicted_label.lower():
                    st.success(f"✅ The plant is Healthy ({confidence:.2f}%)")
                else:
                    st.error(f"❌ The plant is Unhealthy ({confidence:.2f}%)")
            else:
                st.markdown(f"<h4 style='color: #2F3130;'>🩺 {predicted_label} ({confidence:.2f}%)</h4>", unsafe_allow_html=True)

            # Grad-CAM Visualization
            st.subheader("🖼️ Model Focus (Grad-CAM)")
            heatmap = get_gradcam_heatmap(img_array, model)
            superimposed_img = overlay_heatmap(np.array(image.resize((224, 224))), heatmap)
            st.image(superimposed_img, caption='Grad-CAM Heatmap', use_column_width=True)

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
else:
    st.warning("📥 Please upload an image to proceed.")

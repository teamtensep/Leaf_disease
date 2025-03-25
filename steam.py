import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('again_new_scratch')

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

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image = image.resize((224, 224))  # Resize to match model input size
    img_array = np.array(image)

    if st.button('🔍 Predict'):
        st.image(image, caption='Uploaded Image (Resized to 224x224)', use_column_width=True)

        try:
            # Expand dimensions to match model input shape (batch size, height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)

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

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
else:
    st.warning("📥 Please upload an image to proceed.")

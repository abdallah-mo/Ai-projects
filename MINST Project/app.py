import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import os

st.set_page_config(page_title="MNIST Prediction App", page_icon="🔢")
st.title("🔢 MNIST Digit Recognizer")
st.markdown("Draw a digit (0-9) inside the canvas below to test the Convolutional Neural Network.")

@st.cache_resource
def load_keras_model():
    model_path = os.path.join("model", "mnist_model.h5")
    if not os.path.exists(model_path):
        model_path = "mnist_model.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_keras_model()

if model is None:
    st.error("Model not found! Please run the notebook first to generate 'mnist_model.h5'.")
else:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Draw Here")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    
    with col2:
        st.subheader("Prediction Details")
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            if np.sum(gray_img) > 0:
                resized_img = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_AREA)
                normalized_img = resized_img.astype('float32') / 255.0
                preprocessed_img = normalized_img.reshape(1, 28, 28, 1)
                
                prediction_probs = model.predict(preprocessed_img)
                predicted_class = np.argmax(prediction_probs)
                confidence = np.max(prediction_probs) * 100
                
                st.markdown(f"**Predicted Digit:** <h1 style='color: #4CAF50;'>{predicted_class}</h1>", unsafe_allow_html=True)
                st.progress(int(confidence))
                st.write(f"Confidence: **{confidence:.2f}%**")
                
                st.markdown("### Processed Input Image")
                st.image(resized_img, width=150)
            else:
                st.info("Awaiting input...")

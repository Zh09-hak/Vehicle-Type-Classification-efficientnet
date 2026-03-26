import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Заголовок
st.title("🚗 Vehicle Classification AI")

# Загрузка модели
@st.cache_resource
def load_model_cached():
    return load_model("model.keras")

model = load_model_cached()

labels = {
    0: 'Auto Rickshaws',
    1: 'Bikes',
    2: 'Cars',
    3: 'Motorcycles',
    4: 'Planes',
    5: 'Ships',
    6: 'Trains'
}

# Загрузка изображения
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # preprocessing
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_processed = preprocess_input(img_rgb)
    img_processed = np.expand_dims(img_processed, axis=0)

    # prediction
    pred = model.predict(img_processed)
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    st.subheader(f"Prediction: {labels[class_id]}")
    st.write(f"Confidence: {confidence:.4f}")

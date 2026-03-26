import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown
import os

if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    
MODEL_PATH = "model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        url = "https://drive.google.com/uc?1hK9yXRcMD72i94n5dCY7joW3aJVtlAGV"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

st.write("Files in directory:", os.listdir())
    
st.title("🚗 Vehicle Classification AI")

@st.cache_resource
def load_model_cached():
    download_model()
    return load_model(MODEL_PATH)

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

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img = np.array(img)
    
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    st.subheader(f"Prediction: {labels[class_id]}")
    st.write(f"Confidence: {confidence:.4f}")

import streamlit as st
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
from tensorflow.keras.models import load_model
import numpy as np 
from PIL import Image 
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
 

@st.cache_data()
def load():
    model_path = "../best_model_oub_int.h5"
    model = load_model(model_path, compile=False)
    return model

# Chargement du model
model = load()
    
def predict(upload):
    
    img = Image.open(upload)
    img = np.asarray(img) # Transformer l'image en tableau numpy
    img_resize = cv2.resize(img, (224, 224)) # Redimensionner à 224/224
    img_resize = np.expand_dims(img_resize, axis=0)
    pred = model.predict(img_resize)

    rec = pred[0][0]

    return rec


st.title("Poubelle Intelligente")

upload = st.file_uploader("Chargez l'image de votre objet", type=["png", "jpeg", "jpg"])

c1, c2 = st.columns(2)

if upload:
    rec = predict(upload)
    probabilite_recyclable = rec * 100
    probabilite_organic = rec * 100

    c1.image(Image.open(upload))
    if probabilite_recyclable > 50:
        c2.write(f"Je suis certain à {probabilite_recyclable:.2f} % que l'objet est recyclabe")
    else:
        c2.write(f"Je suis certain à {probabilite_organic:.2f} % que l'objet n'est pas recyclabe")


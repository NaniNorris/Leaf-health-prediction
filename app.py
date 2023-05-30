import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st


st.title('Plant leave health prediction app')

# Loading model and defining the classes

MODEL = tf.keras.models.load_model("model_best")
CLASS_NAMES = ["Not healthy","Healthy"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(data))
    return image

up_image = st.file_uploader('Drop your image',type=["jpg", "jpeg", "png"])


def predicts(img):
    image = read_file_as_image(img)
    img_batch = np.expand_dims(image,axis=0)
    prediction = MODEL.predict(img_batch)
    prediction_1 = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.round(np.max(prediction[0])*100,2)

    return predicted_class,confidence,prediction_1,image

if st.button("Predict"):
    classes , confidence , prediction,image = predicts(up_image)
    st.image(image,caption="Image",use_column_width=True)
    if prediction == 1:
        st.success(f"Leaf is {classes}")
        st.success(f"Confidence level = {confidence}%")
    if prediction == 0:
        st.error(f"Leaf is {classes}")
        st.error(f"Confidence level = {confidence}%")
    
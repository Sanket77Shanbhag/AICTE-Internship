import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("potato.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select page', ['Home','Disease Recognition'])

img = Image.open('Diseases.jpeg')
st.image(img)

if app_mode == 'Home':
    st.title("Plant Disease System for Sustainable Agriculture")

if app_mode == 'Disease Recognition':
    st.header("Plant Disease Detection System for Sustainable Agriculture")

    test_image = st.file_uploader("Choose an image...")
    if(st.button("Show Image")):
        st.image(test_image, width=4, use_container_width=True)

    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success("Model is predicting its a {}".format(class_name[result_index]))
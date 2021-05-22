# -*- coding: utf-8 -*-

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import imutils
from PIL import Image
from keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_model_is_mri():
    model = load_model("model_is_mri")
    model.make_predict_function()
    return model

@st.cache(allow_output_mutation=True)
def load_model_classifier():
    model = load_model("model_classifier_brain_tumors")
    model.make_predict_function()
    return model

with st.sidebar:
    text_presentation = components.html(
    """<h1 style="color:#115764; font-size:230%"><b>NEUROLOGS</b><br></h1>
    <hr color=#ffcc00><br>
    <p style="font-size:110%;"><br>Brain disorders are a major public health problem and addressing their enormous social and economic burden is an absolute emergency.<br><br>
    As well as <b>a formidable challenge</b>.<br><br>
    <b>Artificial Intelligence technologies</b> could revolutionize the medicine by providing efficient and relevant tools for innovative therapeutic approaches 
    and improved personalized treatments.</p>""", height=470)

        
title = components.html("""<div><h1 style="color:#9CFF8B; font-size:210%; text-align:center"><b>Brain tumors classifier</b></h1></div>""", height=110)

image_app1 = Image.open("photo/photo_app1.jpg")                        
st.image(image_app1, use_column_width=True)

presentation = components.html("""<div style="color:white; font-size:108%"><br><p>We have designed a deep learning model (convolutional neural network) which can detect and classify 
                               the most common primary brain tumors: glioma, meningioma and pituitary tumors.</p>
                               <p>The model has been trained with more than 3200 MRI scans images and predictions are correct 95% of the time.</p>
                               <p>Please test our medical diagnosis tool: upload a brain scan and obtain the result in only a minute !</p></div>""", height=210)
                       

uploaded_file = st.file_uploader('')
generate_pred = st.button("Prediction")


def crop_image(image):
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    img_thresh = cv2.threshold(img_gray, 45, 255, cv2.THRESH_BINARY)[1]
    img_thresh = cv2.erode(img_thresh, None, iterations=2)
    img_thresh = cv2.dilate(img_thresh, None, iterations=2)

    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]  

    return new_image

col1, col2 = st.beta_columns(2)


if uploaded_file is not None:
    with col1:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        test_image = cv2.imdecode(file_bytes, cv2.IMREAD_ANYCOLOR)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        st.image(test_image, width=300)
        test_image = crop_image(test_image)
        test_image = cv2.resize(test_image, (224,224))
        test_image = np.expand_dims(test_image, axis=0)
    
    if generate_pred:
        with st.spinner('Analysis in progress ......'):
            with col2:
                verif = load_model_is_mri()
                verif_ismri = verif.predict(test_image)
                verif = np.argmax(verif_ismri)
                if verif== 1:
                    components.html("""<div><br>
                                    <p style="background-color:#F63366; text-align:center; font-size:120%; color:white"><br>
                                    Are you sure it is a brain scan image ?<br><br>Please upload another file.<br><br></p>
                                    </div>""", height=200)
                if verif == 0:
                    loaded_model = load_model_classifier()
                    prediction = loaded_model.predict(test_image)
                    prediction = np.argmax(prediction)
                    if prediction == 0:
                        components.html("""<div style="background-color:#99c5c2">
                        <p style="text-align:center; font-size:160%; color:#115764"><br><b>-- RESULT --</b></p>
                        <p style="background-color:#F63366; text-align:center; font-size:140%; color:white"><br><b>Tumor detected</b><br><br></p>
                        <p style="text-align:center; font-size:140%; color:#115764">Classified as <span style="color:red"><b>glioma</b></span> tumor.</p>
                        <br></div>""", height=450)
                    elif prediction == 1:
                        components.html("""<div style="background-color:#99c5c2">
                        <p style="text-align:center; font-size:160%; color:#115764"><br><b>-- RESULT --</b></p>
                        <p style="background-color:#F63366; text-align:center; font-size:140%; color:white"><br><b>Tumor detected</b><br><br></p>
                        <p style="text-align:center; font-size:140%; color:#115764">Classified as <span style="color:navy"><b>meningioma</b></span> tumor.</p>
                        <br></div>""", height=450)
                    elif prediction ==2:
                        components.html("""<div style="background-color:#99c5c2">
                        <p style="text-align:center; font-size:160%; color:#115764"><br><b>-- RESULT --</b></p>
                        <p style="background-color:#9CFF8B; text-align:center; font-size:140%; color:#115764"><br><b>There is no tumor.</b><br><br></p>
                        <br><br></div>""", height=450)
                    elif prediction == 3: 
                        components.html("""<div style="background-color:#99c5c2">
                        <p style="text-align:center; font-size:160%; color:#115764"><br><b>-- RESULT --</b></p>
                        <p style="background-color:#F63366; text-align:center; font-size:140%; color:white"><br><b>Tumor detected</b><br><br></p>
                        <p style="text-align:center; font-size:140%; color:#115764">Classified as <span style="color:green"><b>pituitary</b></span> tumor.</p>
                        <br></div>""", height=450)
                    else:
                        st.write("Error. The algorithm failed to predict the outcome. Please try again")
                      



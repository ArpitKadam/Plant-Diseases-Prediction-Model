import streamlit as st
import tensorflow as tf 
import numpy as np

#OUR MODEL
def model_prediction(test_image):
    model = tf.keras.models.load_model("C:\\Users\\admin\\OneDrive\\Desktop\\PROJECTS\\Plant Diseases Prediction Model\\trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = np.array([input_array])
    prediction = model.predict(input_array)
    result_index = np.argmax(prediction)
    return result_index

#SIDEBAR
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("select page", ["Home", "About", "Diseases Prediction"])

#HOME PAGE
if app_mode == "Home":
    st.balloons()
    st.header("PLANT DISEASES RECOGNITION SYSTEM")
    st.markdown(""" 
# Welcome to our Plant Diseases Recognition System.


We are here to help you with your plant diseases.


## **Our Vision**
Our Vision is to help people with their plant diseases.

## **Our Mission**
Our mission is to help people with their plant diseases.    

## **Our Values**
We are committed to helping people with their plant diseases.

## **Contact us**
Mobile: 123-456-7890 Email: 6HnUH@example.com
If you have any questions or concerns, please contact us.

# **Happy Planting!**
                
""")
    

#ABOUT PAGE
elif app_mode == "About":
    st.snow()
    st.header("About")
    st.markdown("Our team is dedicated to helping people with their plant diseases.")
    st.markdown("We are here to help you with your plant diseases.")
    st.markdown("""
                
      This dataset is recreaded from Kaggle Original dataser link : https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset. The original dataset can also be found on GitHub. The dataset consists of 87K rgb images of plants with their respective labels which can be categorized into 38 different plant classes.   
                
""")
    st.markdown("This project is done by Arpit Kadam, Sanket Jadhav, Ketan Suryavanshi")


#DISEASES PREDICTION PAGE
elif app_mode == "Diseases Prediction":
    st.header("Diseases Prediction")    
    test_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)

#PREDICT BUTTON
    if st.button("Predict") and test_image is not None:
        with st.spinner("Predicting..."):
            result_index = model_prediction(test_image)
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            st.write("Predicted Diseases: ", class_name[result_index])
            st.success("Done!")


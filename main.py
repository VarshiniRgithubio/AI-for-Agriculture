import streamlit as st
import tensorflow as tf  # for load the model
import numpy as np

# Tensorflow Model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model1.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # used to give in batch format (convert single image into batch)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
# three pages we are going to create
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home page
if app_mode == "Home":
    st.header("AI FOR AGRICULTURE")
    image_path = "plant_image.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""Welcome to AI for Agriculture 🌱

    This platform uses advanced **Artificial Intelligence** to help farmers and researchers 
    identify crop diseases quickly and accurately. Simply upload a clear image of your plant's 
    leaves, and our AI model will:
    - Detect possible diseases
    - Suggest preventive or corrective measures
    - Provide general crop care tips

    Our mission is to empower agriculture with technology, reduce crop losses, and 
    improve productivity for farmers worldwide. 🚜🌾
    """)

# About page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is related using offline augmentation from the original dataset.  
    The original dataset can be found online.  

    #### Content
    1. Train (70295 images)  
    2. Validation (17573 images)  
    3. Test (33 images)
    """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        predicted_label = class_name[result_index]
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
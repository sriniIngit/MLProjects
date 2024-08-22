import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from io import BytesIO

# Function to download the model
def download_model(url):
    # Load model from a URL
    #url = 'https://raw.githubusercontent.com/sriniIngit/MLProjects/main/Diabetic_Prediction_Deployment/XGBoost_2_model.pkl'
    
    #response = requests.get(url)
    url = 'https://raw.githubusercontent.com/sriniIngit/MLProjects/Diabetic_Prediction_Deployment/XGBoost_2_model.pkl'
    response = requests.get(url)
    # Check for successful response
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch the file: Status code {response.status_code}")
    if "html" in response.headers["Content-Type"]:
        raise ValueError("The fetched file is not a valid pickle file.")
    model_bytes = BytesIO(response.content)
    # Load the model
    try:
        loaded_model = pickle.load(model_bytes)
    except pickle.UnpicklingError:
        raise ValueError("The file could not be unpickled. Ensure it's a valid pickle file.")
    return loaded_model
       
def get_value(val, my_dict):
    return my_dict.get(val)

# URL of the model
model_url = 'https://github.com/sriniIngit/MLProjects/raw/main/Diabetic_Prediction_Deployment/XGBoost_2_model.pkl'

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])
if app_mode == 'Home':
    st.title('Diabetic Early Detection System:')
    st.image('https://raw.githubusercontent.com/sriniIngit/MLProjects/main/Diabetic_Prediction_Deployment/Cover_Page.jpg')  # Ensure the correct path for your image file
    st.write('App realised by: Team 16 - Konduri Srinivas Rao & Subba Rao')

elif app_mode == 'Prediction':
    st.image('https://raw.githubusercontent.com/sriniIngit/MLProjects/main/Diabetic_Prediction_Deployment/slider-short-3.jpg')  # Ensure the correct path for your image file
    st.subheader('Sir/Mme, You need to fill all necessary information to verify if there is an early warning of a diabetic condition for you based on your inputs!')
    st.sidebar.header("Information about the Respondent:")

    HighBP = st.sidebar.selectbox('High Blood Pressure', ["No", "Yes"])
    HighChol = st.sidebar.selectbox('High Cholesterol', ["No", "Yes"])
    BMI = st.sidebar.selectbox('BMI Category', ["Normal", "Obese", "Overweight"])
    Stroke = st.sidebar.selectbox('Stroke', ["No", "Yes"])
    HeartDiseaseorAttack = st.sidebar.selectbox('Heart Disease or Attack', ["No", "Yes"])
    PhysActivity = st.sidebar.selectbox('Physical Activity', ["No", "Yes"])
    Sex = st.sidebar.selectbox('Sex', ["Male", "Female"])
    Fruits = st.sidebar.selectbox('Fruits Consumption', ["No", "Yes"])
    Veggies = st.sidebar.selectbox('Vegetable Consumption', ["No", "Yes"])
    HvyAlcoholConsump = st.sidebar.selectbox('Heavy Alcohol Consumption', ["No", "Yes"])
    AnyHealthcare = st.sidebar.selectbox('Any Healthcare Access', ["No", "Yes"])
    GenHlth = st.sidebar.selectbox('General Health', ["excellent", "very good", "good", "fair", "poor"])
    MentHlth = st.sidebar.slider('Mental Health (days)', 1, 30, 1)
    PhysHlth = st.sidebar.slider('Physical Health (days)', 1, 30, 1)
    DiffWalk = st.sidebar.selectbox('Difficulty Walking', ["No", "Yes"])
    Age = st.sidebar.selectbox('Age Group', ["Level 1", "Level 2"])
    Education = st.sidebar.selectbox('Education Level', ['Graduate', 'Not Graduate'])

    feature_list = [
        get_value(HighBP, {"No": 0, "Yes": 1}),
        get_value(HighChol, {"No": 0, "Yes": 1}),
        get_value(BMI, {"Normal": 0, "Obese": 1, "Overweight": 2}),
        get_value(Stroke, {"No": 0, "Yes": 1}),
        get_value(HeartDiseaseorAttack, {"No": 0, "Yes": 1}),
        get_value(PhysActivity, {"No": 0, "Yes": 1}),
        get_value(Sex, {"Male": 1, "Female": 0}),
        get_value(Fruits, {"No": 0, "Yes": 1}),
        get_value(Veggies, {"No": 0, "Yes": 1}),
        get_value(HvyAlcoholConsump, {"No": 0, "Yes": 1}),
        get_value(AnyHealthcare, {"No": 0, "Yes": 1}),
        get_value(GenHlth, {"excellent": 1, "very good": 2, "good": 3, "fair": 4, "poor": 5}),
        MentHlth,
        PhysHlth,
        get_value(DiffWalk, {"No": 0, "Yes": 1}),
        get_value(Age, {"Level 1": 1, "Level 2": 2}),
        get_value(Education, {'Graduate': 1, 'Not Graduate': 2}),
    ]

    single_sample = np.array(feature_list).reshape(1, -1)

    if st.button("Predict"):
        #url = 'https://raw.githubusercontent.com/sriniIngit/MLProjects/Diabetic_Prediction_Deployment/XGBoost_2_model.pkl'
        url =       'https://github.com/sriniIngit/MLProjects/blob/main/Diabetic_Prediction_Deployment/XGBoost_2_model.pkl'
        loaded_model = download_model(url)
        model_bytes = BytesIO(response.content)
        # Load the model
        loaded_model = pickle.load(model_bytes)
        # Make prediction
        
        #loaded_model = pickle.load(open('C:/Users/kondu/XGBoost_2_model.pkl', 'rb'))
        prediction = loaded_model.predict(single_sample)

        # Display result
        if prediction[0] == 0:
            st.error('According to our Analysis, you are not at Risk')
        elif prediction[0] == 1:
            st.success('We predict you may have a diabetic condition in the future, please consult a Doctor!')

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

@st.cache_data(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():
        if val == key:
            return value

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction'])

if app_mode == 'Home':
    st.title('Diabetic Early Detection System:')
    
    # Use a local path or a direct URL
    st.image('path/to/Cover_Page.jpg')  # Local path
    # st.image('https://raw.githubusercontent.com/sriniIngit/MLProjects/main/Diabetic_Prediction_Deployment/Cover_Page.jpg')  # URL
    
    st.write('App realised by : Team 16 - Konduri Srinivas Rao & Subba Rao')

elif app_mode == 'Prediction':
    st.image('slider-short-3.jpg')
    st.subheader('Sir/Mme, You need to fill all necessary information in order to verify if there is an early warning of a Diabetic condition for you based on your inputs!')
    
    st.sidebar.header("Information about the Respondent:")
    
    # Define dictionaries
    HighBP = {"No": 0, "Yes": 1}
    HighChol = {"No": 0, "Yes": 1}
    BMI = {"Normal": 0, "Obese": 1, "Overweight": 2}
    Stroke = {"No": 0, "Yes": 1}
    HeartDiseaseorAttack = {"No": 0, "Yes": 1}
    PhysActivity = {"No": 0, "Yes": 1}
    Sex = {"Male": 1, "Female": 0}
    Fruits = {"No": 0, "Yes": 1}
    Veggies = {"No": 0, "Yes": 1}
    HvyAlcoholConsump = {"No": 0, "Yes": 1}
    AnyHealthcare = {"No": 0, "Yes": 1}
    GenHlth = {"excellent": 1, "very good": 2, "good": 3, "fair": 4, "poor": 5}
    DiffWalk = {"No": 0, "Yes": 1}
    Age = {"Level 1": 1, "Level 2": 2}
    Education = {'Graduate': 1, 'Not Graduate': 2}

    # Assuming the following variables are defined based on user input
    Gender = 'Male'  # Example user input
    Married = 'Yes'
    ApplicantIncome = 5000  # Example value
    CoapplicantIncome = 2000  # Example value
    LoanAmount = 200  # Example value
    Loan_Amount_Term = 360.0  # Example value
    Credit_History = 1.0  # Example value
    Property_Area = 'Urban'
    Dependents = '0'
    Self_Employed = 'No'

    # Feature list and prediction
    feature_list = [ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, get_value(Gender, Sex), get_fvalue(Married), get_fvalue(Self_Employed)]
    single_sample = np.array(feature_list).reshape(1, -1)

    if st.button("Predict"):
        file_ = open("6m-rain.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        file = open("green-cola-no.gif", "rb")
        contents = file.read()
        data_url_no = base64.b64encode(contents).decode("utf-8")
        file.close()

        loaded_model = pickle.load(open('Random_Forest.sav', 'rb'))
        prediction = loaded_model.predict(single_sample)

        if prediction[0] == 0:
            st.error('According to our Analysis, you are not at Risk')
            st.markdown(f'<img src="data:image/gif;base64,{data_url_no}" alt="no risk gif">', unsafe_allow_html=True)
        elif prediction[0] == 1:
            st.success('We predict you may have a diabetic condition in future please consult a Doctor!')
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="risk gif">', unsafe_allow_html=True)

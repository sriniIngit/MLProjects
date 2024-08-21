import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
       
app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction'])
if app_mode=='Home':
    st.title('Diabetic Early Detection System :')
    st.image('/sriniIngit/MLProjects/blob/main/Diabetic_Prediction_Deployment/Cover_Page.jpg') 
    st.write('App realised by : Team 16 - KOnduri Srinivas Rao & Subba Rao') 
   
   
   
elif app_mode =='Prediction':
    st.image('slider-short-3.jpg')
    st.subheader('Sir/Mme , You need to fill all neccesary informations in order to verify if there is an early warning of Diabetic condition for you based on your inputs !')
    st.sidebar.header("Informations about the Respondent :")
    HighBP = {"No":0,"Yes":1}
    HighChol = {"No":0,"Yes":1}
    BMI = {"Normal":0,"Obese":1, "Overweight":2}
    Stroke = {"No":0,"Yes":1}
    HeartDiseaseorAttack = {"No":0,"Yes":1}
    PhysActivity = {"No":0,"Yes":1}
    Sex = {"Male":1,"Female":0}
    feature_dict = {"No":1,"Yes":2}
    Fruits = {"No":0,"Yes":1}
    Veggies = {"No":0,"Yes":1}
    HvyAlcoholConsump = {"No":0,"Yes":1}
    AnyHealthcare = {"No":0,"Yes":1}
    GenHlth = {"excellent":1, "very good":2, "good": 3,"fair": 4, "poor":5}
       
    MentHlth = {1:30}
    PhysHlth = {1:30}
    DiffWalk = {"No":0,"Yes":1}
    Age= {"Level 1","Level 2"}
    Education={'Graduate':1,'Not Graduate':2}
    
	#prop={'Rural':1,'Urban':2,'Semiurban':3}
    #Gender=st.sidebar.radio('Gender',tuple(gender_dict.keys()))
    #Married=st.sidebar.radio('Married',tuple(feature_dict.keys()))
    #Self_Employed=st.sidebar.radio('Self Employed',tuple(feature_dict.keys()))
    #Dependents=st.sidebar.radio('Dependents',options=['0','1' , '2' , '3+'])
    #Education=st.sidebar.radio('Education',tuple(edu.keys()))
    #ApplicantIncome=st.sidebar.slider('ApplicantIncome',0,10000,0,)
    #CoapplicantIncome=st.sidebar.slider('CoapplicantIncome',0,10000,0,)
    #LoanAmount=st.sidebar.slider('LoanAmount in K$',9.0,700.0,200.0)
    #Loan_Amount_Term=st.sidebar.selectbox('Loan_Amount_Term',(12.0,36.0,60.0,84.0,120.0,180.0,240.0,300.0,360.0))
    #Credit_History=st.sidebar.radio('Credit_History',(0.0,1.0))
    #Property_Area=st.sidebar.radio('Property_Area',tuple(prop.keys()))


    #class_0 , class_3 , class_1,class_2 = 0,0,0,0
    #if Dependents == '0':
    #    class_0 = 1
    #elif Dependents == '1':
    #    class_1 = 1
    #elif Dependents == '2' :
    #    class_2 = 1
    #else:
    #    class_3= 1

    #Rural,Urban,Semiurban=0,0,0
    #if Property_Area == 'Urban' :
    #    Urban = 1
    #elif Property_Area == 'Semiurban' :
    #    Semiurban = 1
    #else :
    #    Rural=1
   
    data1={
    'Gender':Gender,
    'Married':Married,
    'Dependents':[class_0,class_1,class_2,class_3],
    'Education':Education,
    'ApplicantIncome':ApplicantIncome,
    'CoapplicantIncome':CoapplicantIncome,
    'Self Employed':Self_Employed,
    'LoanAmount':LoanAmount,
    'Loan_Amount_Term':Loan_Amount_Term,
    'Credit_History':Credit_History,
    'Property_Area':[Rural,Urban,Semiurban],
    }

    feature_list=[ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,get_value(Gender,gender_dict),get_fvalue(Married),data1['Dependents'][0],data1['Dependents'][1],data1['Dependents'][2],data1['Dependents'][3],get_value(Education,edu),get_fvalue(Self_Employed),data1['Property_Area'][0],data1['Property_Area'][1],data1['Property_Area'][2]]
    {'HighBP': 1.0958915655402286, 'HighChol': 1.1244044476444752, 'CholCheck': 0.20549903279979453, 'BMI': 1.6662508886561826, 'Smoker': 1.0709079896049265, 'Stroke': -0.21660609138004175, 'HeartDiseaseorAttack': -0.33947806626916344, 'PhysActivity': -1.6570765551903963, 'BMICategory': -0.1617791529740957, 'HealthRiskScore': 1.4455446733710295, 'HealthyLifestyleIndex': -1.483145462253199, 'AgeGroup': 0.2956690615327271, 'ChronicConditionCount': 1.5436647124455432, 'MentalPhysicalHealthScore': 1.8045579506699434, 'HealthcareAccessIssue': -0.28058031756540625}
    
    single_sample = np.array(feature_list).reshape(1,-1)

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
        if prediction[0] == 0 :
            st.error(
    'According to our Analysis, you are not at Risk'
    )
            st.markdown(
    f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">',
    unsafe_allow_html=True,)
        elif prediction[0] == 1 :
            st.success(
    'We predict you may have a diabetic condition in future please consult Doctor !!'
    )
            st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
    )






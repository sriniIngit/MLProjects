import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
logging.info('This is an info message in streamlit')


# Function to download the model 
def download_model(url):
	response = requests.get(url)
	# Check for successful response
	if response.status_code != 200:
		raise ValueError(f"Failed to fetch the file: Status code {response.status_code}")
	if "html" in response.headers["Content-Type"]:
		raise ValueError("The fetched file is not a valid pickle file.")
	#Return the BytesIO object containing the file's content
	return BytesIO(response.content)
	
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
	st.image('https://raw.githubusercontent.com/sriniIngit/MLProjects/main/Diabetic_Prediction_Deployment/Cover_Page.jpg')  # Ensure the correct path for your image file
	st.subheader('Sir/Mme, You need to fill all necessary information to verify if there is an early warning of a diabetic condition for you based on your inputs!')
	st.sidebar.header("Information about the Respondent:")
	HighBP = st.sidebar.selectbox('High Blood Pressure', ["No", "Yes"])
	HighChol = st.sidebar.selectbox('High Cholesterol', ["No", "Yes"])
	CholCheck = st.sidebar.selectbox('Cholesterol check in last 5 years', ["No", "Yes"])
	BMI = st.sidebar.selectbox('BMI Category', ["Normal", "Obese", "Overweight"])
	Smoker = st.sidebar.selectbox('Had the person smoked at least 100 cigarettes in entire life? ', ["No", "Yes"])
	Stroke = st.sidebar.selectbox('Stroke', ["No", "Yes"])
	HeartDiseaseorAttack = st.sidebar.selectbox('Heart Disease or Attack', ["No", "Yes"])
	HealthRiskScore = st.sidebar.selectbox('HealthRiskScore', ["Normal", "Moderate", "High"])
	DiffWalk = st.sidebar.selectbox('Difficulty Walking', ["No", "Yes"])
	ChronicConditionCount =  st.sidebar.selectbox('Chronic Condition With BP Cholesterol_Difficult in Walk',["Normal", "Moderate", "High"])
	PhysActivity = st.sidebar.selectbox('Physical Activity', ["No", "Yes"])
	Sex = st.sidebar.selectbox('Sex', ["Male", "Female"])
	#Fruits = st.sidebar.selectbox('Fruits Consumption', ["No", "Yes"])
	Veggies = st.sidebar.selectbox('Vegetable Consumption', ["No", "Yes"])
	HvyAlcoholConsump = st.sidebar.selectbox('Heavy Alcohol Consumption', ["No", "Yes"])
	AnyHealthcare = st.sidebar.selectbox('Any Healthcare Access', ["No", "Yes"])
	GenHlth = st.sidebar.selectbox('General Health', ["excellent", "very good", "good", "fair", "poor"])
	MentHlth = st.sidebar.slider('Mental Health (days)', 1, 30, 1)
	PhysHlth = st.sidebar.slider('Physical Health (days)', 1, 30, 1)
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
		get_value(Veggies, {"No": 0, "Yes": 1}),
		get_value(HvyAlcoholConsump, {"No": 0, "Yes": 1}),
		get_value(HealthRiskScore, {"Normal":0, "Moderate":1, "High":1}),
		get_value(GenHlth, {"excellent": 1, "very good": 2, "good": 3, "fair": 4, "poor": 5}),
		MentHlth,
		PhysHlth,
		get_value(ChronicConditionCount, {"Normal":0, "Moderate":1, "High":1}),
		get_value(Age, {"Level 1": 1, "Level 2": 0})
	]
	single_sample = np.array(feature_list).reshape(1, -1)
	# Print shape
	# Correct usage
	logging.info("Shape: %s", single_sample.shape)
	logging.info("Contents:\n%s", single_sample)
	logging.info(single_sample)
	if st.button("Predict"):
		url = 'https://raw.githubusercontent.com/sriniIngit/MLProjects/main/Diabetic_Prediction_Deployment/XGBoost_2_model.pkl'
		model_bytes = download_model(url)
		# Load the model
		loaded_model = pickle.load(model_bytes)
		# Assuming 'single_sample' is defined elsewhere in your code
		# Make prediction
		try:
			# Make prediction
			prediction = loaded_model.predict(single_sample)
			# Display result
			length = len(prediction)
			logging.info(length)
			if prediction[0] == 0:
				st.error('According to our Analysis, you are not at Risk')
			elif prediction[0] == 1:
				st.success('We predict you may have a diabetic condition in the future, please consult a Doctor!')
		except Exception as e:
			st.error(f"An error occurred: {e}")

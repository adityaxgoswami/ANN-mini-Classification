import streamlit as st
import numpy as np 
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler , OneHotEncoder, LabelEncoder
import pickle
from tensorflow.keras.models import load_model
model = load_model('model.h5')
print(model.summary())  # Check if the model loads correctly


#loading the scalar , and encodersccc
with open('gender_label.pkl','rb') as file:
    gender_label = pickle.load(file)
   
with open('geo_encoder.pkl','rb') as file:
    geo_encoder = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file) 

st.title("Customer Churn Predictor")

geography=st.selectbox('Geography', geo_encoder.categories_[0])
gender=st.selectbox('Gender', gender_label.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
no_of_product=st.slider('Number of products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member= st.selectbox('Is Active Member', [0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[gender_label.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[no_of_product],
    'HasCrCard': [has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
}
)

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded,columns=geo_encoder.get_feature_names_out(['Geography']))

input_data= pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled = scaler.transform(input_data) 
prediction=model.predict(input_data_scaled) git remote add origin https://github.com/adityaxgoswami/ANN-mini-Classification.git

git push -u origin main

st.write(prediction[0])
if prediction[0]>0.5:
   st.write("Likely to Churn")
else:
    st.write("Not likely to churn")     
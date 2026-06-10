import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pandas as pd
import pickle

model=tf.keras.models.load_model('model.h5')

with open ('ohe_encoder.pkl','rb')as file:
    ohe = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open ('scaler.pkl','rb')as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction')

##user input
geography=st.select_slider('Geography',["France", "Germany", "Spain"])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age= st.slider('Age',18,92)
balance= st.number_input('Balance')
credit_score = st.number_input('credit score')
estimated_salary = st.number_input('Estimated salary')
tenure=st.slider('Tenure',0,10)
num_of_products= st.slider('Number of Producte',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Number',[0,1])

## Prepare the input data
input_data = pd.DataFrame(
    {
    'CreditScore':[credit_score],
    'Geography':[geography],
    'Gender':[gender], 
    'Age':[age],
    'Tenure':[tenure], 
    'Balance':[balance], 
    'NumOfProducts': [num_of_products],
    'HasCrCard':[has_cr_card], 
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],      
}
)

input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

#one-hot emcode 'Geography'
geo_encoded = ohe.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=ohe.get_feature_names_out(['Geography'])
)

#combimr ohe columns wiht input data
input_df=pd.concat([input_data.drop('Geography',axis=1),geo_encoded_df],axis=1)

# scale the input data
input_scaled=scaler.transform(input_df)

# prediction churn
prediction=model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction > 0.5:
    st.write('the customer id likely to churn')
else:
    st.write('the customer is not likely to churn')


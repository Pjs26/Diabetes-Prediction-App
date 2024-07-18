# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 19:52:43 2024

@author: pooja soni
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved data
loaded_model = pickle.load(open('C:/Users/pooja soni/Desktop/Final Year Project/trained_model.sav', 'rb'))

# creating a function for Prediction
def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    #Web page title
    st.title('Diabetes Prediction')
    
    #input data from user			
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose	= st.text_input('Glucose Level')
    BloodPressure = st.text_input('BP Level')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction Value')
    Age	= st.text_input('Age of the Person')
    
    #Prediction 
    diagnosis = ''
    
    #Button for Prediction
    if st.button('Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)    
    
if __name__ == '__main__':
    main( )
    
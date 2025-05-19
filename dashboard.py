import pickle 
import streamlit as st
import pandas as pd

# import dataset
diabetes_df = pd.read_csv('diabetes.csv')

# membaca model
diabetes_model_svm = pickle.load(open('diabetes_model_svm.sav', 'rb'))
diabetes_model_rf = pickle.load(open('diabetes_model_rf.sav', 'rb'))

# judul web

st.title('Aplikasi Prediksi Diabetes')

algoritma = st.selectbox("Pilih Algoritma",options=["Support Vector Machine (SVM)", "Random Forest (RF)"])

with st.expander("Contoh data"):
  st.dataframe(diabetes_df)

# membagi kolom
col1, col2 = st.columns(2)

with col1:
  Pregnancies = st.text_input('Input nilai Pregnancies: ')
  Glucose = st.text_input('Input nilai Glucose: ')
  BloodPressure = st.text_input('Input nilai Blood Pressure: ')
  SkinThickness = st.text_input('Input nilai Skin Thickness: ')

with col2:
  Insulin = st.text_input('Input nilai Insulin: ')
  BMI = st.text_input('Input nilai BMI: ')
  DiabetesPedigreeFunction = st.text_input('Input nilai Diabetes Pedigree Function: ')
  Age = st.text_input('Input nilai Age: ')

# code untuk prediksi
diabetes_diagnosis = ''


# membuat tombol untuk prediksi
if st.button('Test Prediksi Diabetes'):

  if "Support Vector Machine (SVM)" in algoritma:
    diabetes_prediction = diabetes_model_svm.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
  if "Random Forest (RF)" in algoritma:
    diabetes_prediction = diabetes_model_rf.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

  if (diabetes_prediction[0] == 1):
    diabetes_diagnosis = 'Pasien terkena diabetes.'
  else:
    diabetes_diagnosis = 'Pasien tidak terkena diabetes.'
  
  st.success(diabetes_diagnosis)
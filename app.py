import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from funciones import *

st.title('Predicción de lluvia de Australia')

# Carga de pipelines
root_path = 'C:/AA1/MLOps/'
pipeline_clas = load_pipeline_and_model(root_path, 'pipeline_clas.joblib', 'keras_model_clas.h5', 'nn_classifier')
pipeline_reg = load_pipeline_and_model(root_path, 'pipeline_reg.joblib', 'keras_model_reg.h5', 'nn_regressor')

# Interfaz de entrada
col1, col2, col3, col4 = st.columns(4)

with col1:
    date = st.date_input('Date')
    location = st.selectbox('Location', ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport'])
    min_temp = st.number_input('MinTemp', -10.0, 50.0, 20.0)
    if min_temp < -10.0 or min_temp > 50.0:
        st.warning('MinTemp no es correcto. Debe estar entre -10.0 y 50.0.')
    max_temp = st.number_input('MaxTemp', -10.0, 50.0, 30.0)
    if max_temp < -10.0 or max_temp > 50.0:
        st.warning('MaxTemp no es correcto. Debe estar entre -10.0 y 50.0.')
    if min_temp >= max_temp:
        st.warning('MinTemp debe ser menor que MaxTemp.')
    temp_9am = st.number_input('Temp9am', -10.0, 50.0, 20.0)
    if temp_9am < -10.0 or temp_9am > 50.0:
        st.warning('Temp9am no es correcto. Debe estar entre -10.0 y 50.0.')
    temp_3pm = st.number_input('Temp3pm', -10.0, 50.0, 25.0)
    if temp_3pm < -10.0 or temp_3pm > 50.0:
        st.warning('Temp3pm no es correcto. Debe estar entre -10.0 y 50.0.')

with col2:
    wind_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    wind_gust_dir = st.selectbox('WindGustDir', wind_directions)
    wind_gust_speed = st.number_input('WindGustSpeed', 0.0, 150.0, 50.0)
    if wind_gust_speed < 0.0 or wind_gust_speed > 150.0:
        st.warning('WindGustSpeed no es correcto. Debe estar entre 0.0 y 150.0.')
    wind_dir_9am = st.selectbox('WindDir9am', wind_directions)
    wind_dir_3pm = st.selectbox('WindDir3pm', wind_directions)
    wind_speed_9am = st.number_input('WindSpeed9am', 0.0, 150.0, 20.0)
    if wind_speed_9am < 0.0 or wind_speed_9am > 150.0:
        st.warning('WindSpeed9am no es correcto. Debe estar entre 0.0 y 150.0.')
    wind_speed_3pm = st.number_input('WindSpeed3pm', 0.0, 150.0, 20.0)
    if wind_speed_3pm < 0.0 or wind_speed_3pm > 150.0:
        st.warning('WindSpeed3pm no es correcto. Debe estar entre 0.0 y 150.0.')

with col3:
    cloud_9am = st.number_input('Cloud9am', 0.0, 9.0, 4.5)
    if cloud_9am < 0.0 or cloud_9am > 9.0:
        st.warning('Cloud9am no es correcto. Debe estar entre 0.0 y 9.0.')
    cloud_3pm = st.number_input('Cloud3pm', 0.0, 9.0, 4.5)
    if cloud_3pm < 0.0 or cloud_3pm > 9.0:
        st.warning('Cloud3pm no es correcto. Debe estar entre 0.0 y 9.0.')
    humidity_9am = st.number_input('Humidity9am', 0.0, 100.0, 50.0)
    if humidity_9am < 0.0 or humidity_9am > 100.0:
        st.warning('Humidity9am no es correcto. Debe estar entre 0.0 y 100.0.')
    humidity_3pm = st.number_input('Humidity3pm', 0.0, 100.0, 50.0)
    if humidity_3pm < 0.0 or humidity_3pm > 100.0:
        st.warning('Humidity3pm no es correcto. Debe estar entre 0.0 y 100.0.')
    pressure_9am = st.number_input('Pressure9am', 900.0, 1100.0, 1015.0)
    if pressure_9am < 900.0 or pressure_9am > 1100.0:
        st.warning('Pressure9am no es correcto. Debe estar entre 900.0 y 1100.0.')
    pressure_3pm = st.number_input('Pressure3pm', 900.0, 1100.0, 1015.0)
    if pressure_3pm < 900.0 or pressure_3pm > 1100.0:
        st.warning('Pressure3pm no es correcto. Debe estar entre 900.0 y 1100.0.')

with col4:
    evaporation = st.number_input('Evaporation', 0.0, 200.0, 5.0)
    if evaporation < 0.0 or evaporation > 200.0:
        st.warning('Evaporation no es correcto. Debe estar entre 0.0 y 200.0.')
    rain_today = st.selectbox('RainToday', ['No', 'Yes'])
    rainfall = st.number_input('Rainfall', 0.0, 500.0, 0.0)
    if rainfall < 0.0 or rainfall > 500.0:
        st.warning('Rainfall no es correcto. Debe estar entre 0.0 y 500.0.')
    sunshine = st.number_input('Sunshine', 0.0, 14.5, 7.5)
    if sunshine < 0.0 or sunshine > 15.0:
        st.warning('Sunshine no es correcto. Debe estar entre 0.0 y 14.5.')

# Preparar datos para la predicción
df_pred = pd.DataFrame({
    'Date': [date],
    'Location': [location],
    'MinTemp': [min_temp],
    'MaxTemp': [max_temp],
    'Rainfall': [rainfall],
    'Evaporation': [evaporation],
    'Sunshine': [sunshine],
    'WindGustDir': [wind_gust_dir],
    'WindGustSpeed': [wind_gust_speed],
    'WindDir9am': [wind_dir_9am],
    'WindDir3pm': [wind_dir_3pm],
    'WindSpeed9am': [wind_speed_9am],
    'WindSpeed3pm': [wind_speed_3pm],
    'Humidity9am': [humidity_9am],
    'Humidity3pm': [humidity_3pm],
    'Pressure9am': [pressure_9am],
    'Pressure3pm': [pressure_3pm],
    'Cloud9am': [cloud_9am],
    'Cloud3pm': [cloud_3pm],
    'Temp9am': [temp_9am],
    'Temp3pm': [temp_3pm],
    'RainToday': [rain_today],
    'RainTomorrow': [1],
    'RainfallTomorrow': [1]
})

df_pred = input_data_preparation (df_pred)
prediccion_clas = pipeline_clas.predict(df_pred)
prediccion_reg = pipeline_reg.predict(df_pred)

# Ajuste de la predicción de RainfallTomorrow
rainfall_tomorrow_prediction = int(max(0, round(prediccion_reg[0])) if prediccion_clas[0] == 1 else 0)

st.sidebar.header('Valores de las predicciones:')
st.sidebar.write('RainTomorrow')
prediction_text = 'Yes' if prediccion_clas[0] == 1 else 'No'
st.sidebar.write(prediction_text)

st.sidebar.write('RainfallTomorrow [mm]')
st.sidebar.text(rainfall_tomorrow_prediction)
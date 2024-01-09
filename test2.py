import streamlit as st
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def load_model():
 model = pickle.load(open('model.pkl', 'rb'))
 return model

model = load_model()

# Initialize input_values in session_state
if 'input_values' not in st.session_state:
 st.session_state.input_values = [1]*15

# Define functions to change input_values
def change_input_values_to_inp1():
 st.session_state.input_values = [1]*15

def change_input_values_to_inp2():
 st.session_state.input_values = [2]*15

# Create buttons to change input_values
st.button("Change to inp1", on_click=change_input_values_to_inp1)
st.button("Change to inp2", on_click=change_input_values_to_inp2)

with st.form(key='my_form'):
 st.title('Price Prediction Model')
 x_min = [1.25,1.0,1.0,800.0,0.0,0.0,0.0,1.0,0.0,0.0,35.0,2.0,2155293.9162760093,96.0, 0]
 x_max = [7.3,200.0,64.0,7500.0,1.0,1.0,1.0,3.0,5.0,1.0,95.0,5.0,15062544.5954953525,3840.0, 0]

 main_row = st.columns((1.5, 1.5))

 features = ['Display Size', 'Rear Camera Pixel', 'Front Camera Pixel',
      'Battery Capacity(mAh)', 'Wireless Charging', '3.5mm Headphone Port',
      'NFC', 'Item Condition', 'price_vnd', 'Material Score', 'pse',
      'CPU Score', 'Highest Technology', 'Brand Target Encoded',
      'Length of Resolution']

 def scale(x, min, max):
  x_scaled = x * (max - min) + min
  return x_scaled

 defauls_value = [6.0,12.0,12.0,3300.0,0.0,1.0,0.0,2.0,4.0,0.0,80.0,4.0,5671592.0083675822,2160.0, 0]
 
 num_features = 15

 # Loop through the range of num_features
 for i in range(num_features):
  input_value = main_row[0].text_input(f'Enter value for feature {features[i]} (inch):', st.session_state.input_values[i])
  input_value = float(input_value)
  st.session_state.input_values[i] = input_value

 submit_button = st.form_submit_button(label='Predict Price')

 if submit_button:
  if None not in st.session_state.input_values:
    np_arr = np.array(st.session_state.input_values)
    np_arr = np_arr.reshape(1, 15)

    prediction = model.predict(np_arr) # replace this with your model's prediction
    with main_row[1].empty():
      st.write('The predicted price is: ', prediction[0])
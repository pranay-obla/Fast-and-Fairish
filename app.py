import streamlit as st
import numpy as np
from keras.models import load_model
import joblib

# Load model
try:
    model = joblib.load('best.pkl')
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Prediction function
def predict(inputs):
    try:
        prediction = model.predict(inputs)
        return prediction[0][0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# One-hot encode categorical features
def one_hot_encode(vehicle_type):
    # Mapping of categories to indices
    category_mapping = {
        'Motorcycle': 0,
        'Car': 1,
        'Sedan': 2,
        'SUV': 3,
        'Van': 4,
        'Bus': 5,
        'Truck': 6
    }
    # Initialize one-hot encoded array
    encoded_array = np.zeros((1, len(category_mapping)))
    # Set the corresponding index to 1
    encoded_array[0, category_mapping[vehicle_type]] = 1
    return encoded_array

# Streamlit UI
st.title('Real-time Prediction')
st.write('Enter information to get predictions:')

# Inputs
col1, col2 = st.columns([1, 2])
with col1:
    vehicle_type = st.selectbox('Vehicle Type', [''] + ['Motorcycle', 'Car', 'Sedan', 'SUV', 'Van', 'Bus', 'Truck'])
    fastag_id = st.text_input('FASTAG ID')
    transaction_amount = st.number_input('Transaction Amount', step=5)
    geographical_location = st.text_input('Geographical Location')
with col2:
    vehicle_plate = st.text_input('Vehicle Plate')
    toll_booth_id = st.text_input('Toll Booth ID (optional)')
    paid_amount = st.number_input('Paid Amount', step=5)
    vehicle_speed = st.text_input('Vehicle Speed (optional)')

lane_type = st.selectbox('Lane Type', [''] + ['Regular', 'Express'])
vehicle_dimensions = st.selectbox('Vehicle Dimensions (optional)', [''] + ['Small', 'Medium', 'Large'])

# Predict button
if st.button('Predict'):
    # Preprocess input data
    # Replace 'vehicle_type' with the actual input data
    vehicle_type_encoded = one_hot_encode(vehicle_type)
    transaction_amount_scaled = transaction_amount
    paid_amount_scaled = paid_amount
    # Concatenate input features
    inputs = np.concatenate([vehicle_type_encoded, [[transaction_amount_scaled, paid_amount_scaled]]], axis=1)
    
    # Make prediction
    prediction = model.predict(inputs)
    st.write('Prediction:', prediction)

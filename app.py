import streamlit as st
import numpy as np
from keras.models import load_model

# Load model
try:
    model = load_model('neural_networks.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Prediction function
def predict(inputs):
    try:
        prediction = model.predict(inputs)
        return prediction[0][0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Streamlit UI
st.title('Real-time Prediction')
st.write('Enter information to get predictions:')

# Inputs
col1, col2 = st.columns([1, 2])
with col1:
    vehicle_type = st.selectbox('Vehicle Type', ['None'] + ['Motorcycle', 'Car', 'Sedan', 'SUV', 'Van', 'Bus', 'Truck'])
    fastag_id = st.text_input('FASTAG ID')
    transaction_amount = st.number_input('Transaction Amount', step=5)
    geographical_location = st.text_input('Geographical Location (Lat, Long)')
with col2:
    vehicle_plate = st.text_input('Vehicle Plate')
    toll_booth_id = st.text_input('Toll Booth ID (optional)')
    paid_amount = st.number_input('Paid Amount', step=5)
    vehicle_speed = st.text_input('Vehicle Speed (optional)')

lane_type = st.selectbox('Lane Type', ['None'] + ['Regular', 'Express'])
vehicle_dimensions = st.selectbox('Vehicle Dimensions (optional)', ['None'] + ['Small', 'Medium', 'Large'])

# Predict button
if st.button('Predict'):
    # Initialize input data with required fields
    inputs = [
        vehicle_type,
        transaction_amount,
        paid_amount,
        geographical_location
    ]
    
    # Append optional fields if they are not empty or 'None'
    optional_fields = [
        vehicle_plate,
        fastag_id,
        toll_booth_id,
        lane_type,
        vehicle_dimensions,
        vehicle_speed
    ]
    for field in optional_fields:
        if field != '' and field != 'None':
            inputs.append(field)
    
    # Ensure that at least the required input fields are present
    if len(inputs) >= 4:
        inputs = np.array([inputs])
        prediction = predict(inputs)
        st.write('Prediction:', prediction)
    else:
        st.error('Please provide at least the required input fields.')

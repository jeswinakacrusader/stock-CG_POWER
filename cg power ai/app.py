import streamlit as st
import torch
import torch.nn as nn
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the neural network model
model_nn = SimpleNN()
model_nn.load_state_dict(torch.load('final_neural_network_model.pth', map_location=torch.device('cpu')))
model_nn.eval()

# Define feature names and their typical ranges (adjust as needed)
features = {
    'Open Price': (0, 1000),
    'Close Price': (0, 1000),
    'High Price': (0, 1000),
    'Low Price': (0, 1000),
    'Volume': (0, 10000000),
    'SMA 10': (0, 1000),
    'SMA 50': (0, 1000),
    'RSI': (0, 100),
    'P/E Ratio': (0, 100),
    'Market Cap': (0, 1000000000000)
}

# Normalization function
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Denormalization function
def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# Define Streamlit app
st.title('Stock Prediction Model')

# Input form
with st.form(key='input_form'):
    st.subheader('Enter the features')
    
    # Input fields
    input_values = {}
    for feature, (min_val, max_val) in features.items():
        try:
            # Ensure min_val and max_val are numbers
            min_val = float(min_val)
            max_val = float(max_val)

            # Print for debugging purposes
            print(f"Feature: {feature}, Min: {min_val}, Max: {max_val}")

            input_values[feature] = st.number_input(
                feature,
                value=(min_val + max_val) / 17,
                min_value=min_val,
                max_value=max_val
            )
        except Exception as e:
            st.error(f"Error with feature '{feature}': {e}")

    submit_button = st.form_submit_button(label='Predict')

# Prediction
if submit_button:
    # Normalize inputs
    normalized_features = [normalize(input_values[feature], *range_vals) for feature, range_vals in features.items()]
    
    # Neural Network Prediction
    features_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model_nn(features_tensor).squeeze().item()
    
    st.subheader(f'Raw Model Output: {prediction:.4f}')
    
    # Interpret the prediction
   # Interpret the prediction
if 'Close Price' in features:
    close_price_range = features['Close Price']
    interpreted_prediction = denormalize(max(0, abs(prediction)), *close_price_range)  # Ensure non-negative
    st.subheader(f'Interpreted Prediction (Close Price): {interpreted_prediction:.2f}')
        
        # Calculate percentage change
    current_close = input_values['Close Price']
    percent_change = ((interpreted_prediction - current_close) / current_close) * 100
    change_direction = "increase" if percent_change > 0 else "decrease"
    st.write(f"This suggests a {abs(percent_change):.2f}% {change_direction} from the current close price.")
else:
        st.write("Unable to interpret prediction as Close Price. Please ensure 'Close Price' is included in the features.")
    
st.write("Note: This interpretation assumes the model is predicting the next closing price. Adjust the interpretation based on what your model was actually trained to predict.")

st.sidebar.write("### Model Information")
st.sidebar.write("This model uses a simple neural network to predict stock behavior based on the input features. The exact prediction target depends on how the model was trained.")
st.sidebar.write("If you're seeing unexpected results, consider:")
st.sidebar.write("1. Retraining the model with normalized data")
st.sidebar.write("2. Adding a final activation function (e.g., ReLU or Sigmoid) to the model architecture")
st.sidebar.write("3. Verifying that the input ranges match those used during training")
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Dry Bean Classifier",
    layout="wide"
)

@st.cache_resource
def load_assets():
    try:
        with open('svm_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('power_transformer.pkl', 'rb') as file:
            transformer = pickle.load(file)
        with open('label_encoder.pkl', 'rb') as file:
            encoder = pickle.load(file)
        return model, scaler, transformer, encoder
    except FileNotFoundError:
        return None, None, None, None
model, scaler, transformer, encoder = load_assets()

st.title("Dry Bean Species Classifier")
st.markdown("Enter the physical measurements of a dry bean to predict its species. This model is built by Ayush Anand.")
if not all([model, scaler, transformer, encoder]):
    st.error("Critical Error: One or more required asset files (.pkl) are missing. Please ensure 'svm_model.pkl', 'scaler.pkl', 'power_transformer.pkl', and 'label_encoder.pkl' are in the same directory as this script.")
else:
    st.header("Bean Measurements")
    st.markdown("Use the input boxes below to enter the bean's features. The fields are pre-filled with the average values from the dataset.")
    feature_limits = {
        'Area':            {'min': 20000.0, 'max': 255000.0, 'value': 28395.00, 'step': 1.0, 'format': "%.2f"},
        'Perimeter':       {'min': 520.0,   'max': 2000.0,   'value': 610.30,   'step': 1.0, 'format': "%.2f"},
        'MajorAxisLength': {'min': 180.0,   'max': 740.0,    'value': 208.18,   'step': 0.1, 'format': "%.2f"},
        'MinorAxisLength': {'min': 120.0,   'max': 465.0,    'value': 173.15,   'step': 0.1, 'format': "%.2f"},
        'AspectRation':    {'min': 1.0,     'max': 2.5,      'value': 1.20,     'step': 0.01, 'format': "%.2f"},
        'Eccentricity':    {'min': 0.2,     'max': 1.0,      'value': 0.55,     'step': 0.01, 'format': "%.2f"},
        'ConvexArea':      {'min': 20500.0, 'max': 263500.0, 'value': 28715.1, 'step': 1.0, 'format': "%.2f"},
        'EquivDiameter':   {'min': 160.0,   'max': 570.0,    'value': 190.15,   'step': 0.1, 'format': "%.2f"},
        'Extent':          {'min': 0.5,     'max': 0.9,      'value': 0.76,     'step': 0.01, 'format': "%.2f"},
        'Solidity':        {'min': 0.9,     'max': 1.0,      'value': 0.99,     'step': 0.01, 'format': "%.2f"},
        'roundness':       {'min': 0.45,    'max': 1.0,      'value': 0.96,     'step': 0.01, 'format': "%.2f"},
        'Compactness':     {'min': 0.6,     'max': 1.0,      'value': 0.91,     'step': 0.01, 'format': "%.2f"},
        'ShapeFactor1':    {'min': 0.002,   'max': 0.011,    'value': 0.0073,   'step': 0.0001, 'format': "%.4f"},
        'ShapeFactor2':    {'min': 0.0005,  'max': 0.004,    'value': 0.0031,   'step': 0.0001, 'format': "%.4f"},
        'ShapeFactor3':    {'min': 0.4,     'max': 1.0,      'value': 0.83,     'step': 0.01, 'format': "%.2f"},
        'ShapeFactor4':    {'min': 0.9,     'max': 1.0,      'value': 1.0,     'step': 0.01, 'format': "%.2f"}
    }
    feature_names = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
        'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
        'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
        'ShapeFactor3', 'ShapeFactor4'
    ]

    cols = st.columns(4)
    user_inputs = []
    for i, feature_name in enumerate(feature_names):
        with cols[i % 4]:
            limits = feature_limits[feature_name]
            value = st.number_input(
                label=feature_name,
                min_value=limits['min'],
                max_value=limits['max'],
                value=limits['value'],
                step=limits['step'],
                format=limits['format'],
                key=feature_name
            )
            user_inputs.append(value)
    if st.button("Predict Bean Type", type="primary"):
        input_array = np.array(user_inputs).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        transformed_input = transformer.transform(scaled_input)
        prediction_encoded = model.predict(transformed_input)
        prediction_label = encoder.inverse_transform(prediction_encoded)[0]

        st.subheader("Prediction Result")
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
             st.image("beans.jpg", width=150)
        with col2:
            st.success(f"**The predicted bean species is:**")
            st.markdown(f"## {prediction_label.title()}")
            st.info("This prediction is based on a Support Vector Machine (SVM) model trained on over 10k samples.")
            
st.markdown("Ayush Anand, MTech, IIT HYDERABAD")

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

# Sidebar for selection
with st.sidebar:
    st.markdown("# CO2 Emission Prediction")
    user_input = st.selectbox('Please select', ('Model',))

# Load data
df = pd.read_csv('co2 Emissions.csv')

# Mapping fuel types
fuel_type_mapping = {"Z": "Premium Gasoline", "X": "Regular Gasoline", "D": "Diesel", "E": "Ethanol(E85)", "N": "Natural Gas"}
df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)
df_natural = df[~df["Fuel Type"].str.contains("Natural Gas")].reset_index(drop=True)

# Selecting relevant features and removing outliers
df_new = df_natural[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
df_new_model = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]

# Model Prediction
if user_input == 'Model':
    st.title("Predict CO2 Emissions")

    # Input sliders for prediction
    engine_size = st.slider('Engine Size (L)', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    cylinders = st.slider('Cylinders', min_value=2, max_value=16, value=4, step=1)
    fuel_consumption_comb = st.slider('Fuel Consumption Comb (L/100 km)', min_value=1.0, max_value=35.0, value=8.5, step=0.1)

    # Gradient Boosting Regressor model
    X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = df_new_model['CO2 Emissions(g/km)']
    model = GradientBoostingRegressor()
    model.fit(X, y)

    # Predict the CO2 Emissions
    input_data = np.array([[engine_size, cylinders, fuel_consumption_comb]])
    prediction = model.predict(input_data)
    st.subheader(f'Predicted CO2 Emissions: {prediction[0]:.2f} g/km')

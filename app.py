import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

st.title("🏠 House Price Prediction")

longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
total_bedrooms = st.number_input("Total Bedrooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

if st.button("Predict Price"):

    data = pd.DataFrame({
        "longitude":[longitude],
        "latitude":[latitude],
        "housing_median_age":[housing_median_age],
        "total_rooms":[total_rooms],
        "total_bedrooms":[total_bedrooms],
        "population":[population],
        "households":[households],
        "median_income":[median_income],
        "ocean_proximity":[ocean_proximity]
    })

    data_prepared = pipeline.transform(data)

    prediction = model.predict(data_prepared)

    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}"
    )

# --- imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

# --- dataset link --- 
# https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset

# --- streamlit initial setup ---
st.set_page_config(
    page_title="Australian Housing Dashboard 🏡",
    layout="centered",
    page_icon="🏡",
)

st.sidebar.title("Australian Housing Dashboard 🏡")
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Introduction 📘",
        "Data Visualization 📊",
        "Prediction 🔮",
        "Explainability 🔍",
        "Hyperparameter Tuning 📈",
        "Conclusion 📖",
    ],
)

st.image("house.jpg", width=500)

df = pd.read_csv("house_pricing_dataset.csv")
df = df.drop(columns=["Id"])

# --- introduction ---
if page == "Introduction 📘":
    st.header("01 Introduction 📘")

    st.subheader("Business Case Presentation 💼")
    st.write("The real estate market relies on accurate property valuations to support " \
    "buying, selling, and investment decisions. This application uses historical housing " \
    "data and machine learning to predict a home's sale price based on factors such as size, " \
    "location, and property characteristics. By providing fast, data-driven estimates, the app " \
    "helps reduce uncertainty and supports more informed decision-making for real estate " \
    "professionals and homeowners.")

    st.markdown("---")

    st.subheader("Data Presentation 👨🏻‍🏫")

    st.markdown("#### Data Preview 🔭")
    rows = st.slider("Select a number of rows to display", 5, 2000)
    st.dataframe(df.head(rows))

    st.markdown("#### Missing values ❓")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")

    st.markdown("#### Summary Statistics 📈")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

# --- visualization ---
elif page == "Data Visualization 📊":
    st.header("02 Data Visualization 📊")
    







    

# --- prediction ---
elif page == "Prediction 🔮":
    st.header("03 Prediction 🔮")


# --- explainability page ---
elif page == "Explainability 🔍":
    st.header("04 Explainability 🔍")
    

# --- MLflow Runs Page ---
elif page == "Hyperparameter Tuning 📈":
    st.header("05 Hyperparameter Tuning 📈")
    

# --- conclusion ---
elif page == "Conclusion 📖":
    st.header("06 Conclusion 📖")


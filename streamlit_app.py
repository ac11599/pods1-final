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

# --- introduction ---
if page == "Introduction 📘":
    st.header("01 Introduction 📘")

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


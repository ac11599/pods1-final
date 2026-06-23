# --- imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    # Axis options
    numeric_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Price']
    categorical_cols = ['Location', 'Condition', 'Garage']

    tab1, tab2, tab3 = st.tabs(
        ["Average Trends Over Time 📈", "Distribution Explorer 📊", "Correlation Heatmap 🔥"])

    with tab1:
        st.header("Average Trends Over Time 📈")
        y_axis = st.selectbox("Select metric to plot over time",
                              numeric_cols, index=3, key="tab1_y")
        col1, col2, col3 = st.columns(3)

        with col1:
            location_filter = st.multiselect("Location", options=["Downtown", "Suburban", "Urban", "Rural"], default=["Downtown", "Suburban", "Urban", "Rural"], key="tab1_location")
        with col2:
            condition_filter = st.multiselect("Condition", options=["Poor", "Fair", "Good", "Excellent"], default=["Poor", "Fair", "Good", "Excellent"], key="tab1_condition")
        with col3:
            garage_filter =  st.multiselect("Garage", options=["Yes", "No"], default=["Yes", "No"], key="tab1_garage")

        filtered_df = df[
            df['Location'].isin(location_filter) &
            df['Condition'].isin(condition_filter) &
            df['Garage'].isin(garage_filter)
        ]

        filtered_df['YearBuilt'] = pd.to_datetime(filtered_df['YearBuilt'], format='%Y')
        time_data = filtered_df.groupby('YearBuilt')[y_axis].mean()
        st.line_chart(time_data)

    with tab2:
        st.header("Distribution Explorer 📊")

        view = st.selectbox("Select view", options=["Numeric", "Categorical"], key="tab2_view")

        if view == "Numeric":
            col1, col2, col3 = st.columns(3)
            with col1:
                location_filter2 = st.multiselect("Location", options=["Downtown", "Suburban", "Urban", "Rural"], default=["Downtown", "Suburban", "Urban", "Rural"], key="tab2_location")
            with col2:
                condition_filter2 = st.multiselect("Condition", options=["Poor", "Fair", "Good", "Excellent"], default=["Poor", "Fair", "Good", "Excellent"], key="tab2_condition")
            with col3:
                garage_filter2 = st.multiselect("Garage", options=["Yes", "No"], default=["Yes", "No"], key="tab2_garage")

            filtered_df2 = df[
                df['Location'].isin(location_filter2) &
                df['Condition'].isin(condition_filter2) &
                df['Garage'].isin(garage_filter2)
            ]

            st.subheader("Numeric Distributions")
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.hist(filtered_df2[col], bins=30, color='steelblue', edgecolor='white')
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                st.pyplot(fig)
        
        elif view == "Categorical":
            st.subheader("Categorical Distributions")
            for col in categorical_cols:
                fig, ax = plt.subplots(figsize=(10, 3))
                counts = df[col].value_counts()
                ax.bar(counts.index, counts.values, color='steelblue', edgecolor='white')
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                st.pyplot(fig)

    with tab3:
        st.header("Correlation Heatmap 🔥")
        df_numeric = df[numeric_cols]
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_numeric.corr(), annot=True,
                    fmt=".2f", cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

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


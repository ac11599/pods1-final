# --- imports ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# --- dataset link --- 
# https://www.kaggle.com/datasets/sukhmandeepsinghbrar/house-Prices-india

# --- streamlit initial setup ---
st.set_page_config(
    page_title="Indian Housing Dashboard 🏡",
    layout="centered",
    page_icon="🏡",
)

st.sidebar.title("Indian Housing Dashboard 🏡")
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

df = pd.read_csv("House Price India.csv")
df = df.drop(columns=["id", "Date", "Lattitude", "Longitude"])

# --- introduction ---
if page == "Introduction 📘":
    st.header("01 Introduction 📘")

    st.subheader("Business Case Presentation 💼")
    st.write("The real estate market relies on accurate property valuations to support " \
    "buying, selling, and investment decisions. This application uses historical housing " \
    "data and machine learning to predict a home's sale Price based on factors such as size, " \
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

    numeric_cols = ['Price', 'living area', 'lot area', 'Area of the house(excluding basement)', 
                    'Area of the basement', 'living_area_renov', 'lot_area_renov',
                    'number of bedrooms', 'number of bathrooms', 'number of floors',
                    'grade of the house', 'condition of the house', 'number of views',
                    'Number of schools nearby', 'Distance from the airport']
    categorical_cols = ['waterfront present', 'was_renovated']
    df['was_renovated'] = (df['Renovation Year'] != 0).astype(int)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distribution Explorer 🏠", "Price Analysis 💰", "Trends Over Time 📈", "Correlation Heatmap 🔥"])
    
    # --- tab 1: distribution explorer ---
    with tab1:
        st.header("Distribution Explorer 🏠")

        view = st.selectbox("Select view", options=["Numeric", "Categorical"], key="tab2_view")

        if view == "Numeric":
            st.subheader("Numeric Distributions")
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.hist(df[col], bins=30, edgecolor='white')
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                plt.tight_layout()
                st.pyplot(fig)

        elif view == "Categorical":
            st.subheader("Categorical Distributions")
            for col in categorical_cols:
                fig, ax = plt.subplots(figsize=(10, 3))
                counts = df[col].value_counts()
                ax.bar(counts.index, counts.values,edgecolor='white')
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                plt.tight_layout()
                st.pyplot(fig)

    # --- tab 2: price analysis ---
    with tab2:
        st.header("Price Analysis 💰")

        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df['Price'], bins=50, edgecolor='white')
        ax.set_xlabel("Price")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Price")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        st.subheader("Renovated vs Non-Renovated Average Price")
        fig, ax = plt.subplots(figsize=(10, 4))
        renov_data = df.groupby('was_renovated')['Price'].mean()
        ax.bar(['Not Renovated', 'Renovated'], renov_data.values, edgecolor='white')
        ax.set_xlabel("Renovation Status")
        ax.set_ylabel("Average Price")
        ax.set_title("Average Price: Renovated vs Not Renovated")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        x_axis = st.selectbox("Select metric", numeric_cols, key="tab1_x")

        st.subheader(f"Price vs {x_axis}")
        fig, ax = plt.subplots(figsize=(10, 4))
        if df[x_axis].nunique() > 20:
            ax.scatter(df[x_axis], df['Price'], alpha=0.5)
            ax.set_ylabel("Price")
            ax.set_title(f"Price vs {x_axis}")
        else:
            data = df.groupby(x_axis)['Price'].mean()
            ax.bar(data.index, data.values, edgecolor='white', width=0.5)
            ax.set_ylabel("Average Price")
            ax.set_title(f"Average Price by {x_axis}")
        ax.set_xlabel(f"{x_axis}")
        plt.tight_layout()
        st.pyplot(fig)
        
    # --- tab 3: trends over time ---
    with tab3:
        st.header("Trends Over Time 📈")

        y_axis = st.selectbox("Select metric", numeric_cols, key="tab3_y")

        fig, ax = plt.subplots(figsize=(10, 5))
        time_data = df.groupby('Built Year')[y_axis].mean().reset_index()
        ax.plot(time_data['Built Year'], time_data[y_axis], marker='o', linewidth=2)
        ax.set_xlabel("Built Year")
        ax.set_ylabel(f"Average {y_axis}")
        ax.set_title(f"Average {y_axis} by Built Year")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}s"))
        plt.tight_layout()
        st.pyplot(fig)

    # --- tab 4: correlation heatmap ---
    with tab4:
        st.header("Correlation Heatmap 🔥")
        df_numeric = df[numeric_cols]
        fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
        sns.heatmap(df_numeric.corr(numeric_only=True), annot=True,
                    fmt=".2f", cmap='coolwarm', ax=ax_corr)
        plt.tight_layout()
        st.pyplot(fig_corr)

# --- prediction ---
elif page == "Prediction 🔮":
    st.header("03 Prediction 🔮")

    # feature selection
    features = df.columns.tolist()
    features.remove("Price")
    features_selection = st.sidebar.multiselect("Select Features (X)", features, default=features)

    # model choice
    model_name = st.sidebar.selectbox(
        "Choose Model",
        ["Linear Regression", "Decision Tree", "Random Forest"],
    )

    # parameter settings
    params = {}
    if model_name == "Decision Tree":
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "Random Forest":
        params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)

    selected_metrics = st.sidebar.multiselect(
        "Metrics to display",
        ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"],
        default=["Mean Absolute Error (MAE)", "Mean Absolute Error (MAE)", "R² Score"],
    )

    # data preparation, fitting the model, training, and prediction
    X = df[features_selection]
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(**params, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(**params, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # metrics
    mse = metrics.mean_squared_error(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)

    st.write(f"**MSE:** {mse:,.2f}")
    st.write(f"**MAE:** {mae:,.2f}")
    st.write(f"**R² Score:** {r2:.3f}")

# --- explainability page ---
elif page == "Explainability 🔍":
    st.header("04 Explainability 🔍")
    

# --- hyperparamter tuning Page ---
elif page == "Hyperparameter Tuning 📈":
    st.header("05 Hyperparameter Tuning 📈")
    

# --- conclusion ---
elif page == "Conclusion 📖":
    st.header("06 Conclusion 📖")


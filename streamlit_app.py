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

    st.markdown("""
    ### 🎯 What is this page?
    Here we explain why the model makes the predictions it does.
    We look at which features (columns) have the strongest influence on house price.
    This helps answer the question: *what actually makes a house more expensive?*
    """)
    st.markdown("---")

    df_exp = df.copy()
    df_exp['was_renovated'] = (df_exp['Renovation Year'] != 0).astype(int)
    df_exp = df_exp.drop(columns=['Renovation Year'])

    features_exp = df_exp.columns.tolist()
    features_exp.remove("Price")
 
    X_exp = df_exp[features_exp]
    y_exp = df_exp["Price"]
 
    X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
        X_exp, y_exp, test_size=0.2, random_state=42
    )

    model_choice = st.selectbox(
        "Select model to explain",
        ["Linear Regression", "Decision Tree", "Random Forest"]
    )
 
    if model_choice == "Linear Regression":
        model_exp = LinearRegression()
        model_exp.fit(X_train_exp, y_train_exp)
        coef_df = pd.DataFrame({
            "Feature": features_exp,
            "Coefficient": model_exp.coef_
        }).sort_values("Coefficient", ascending=False)
 
        st.subheader("📊 Feature Coefficients — Linear Regression")
        st.markdown("""
        Each bar shows how much the predicted price changes when that feature increases by 1 unit.
        - 🟢 **Green** = pushes price **up**
        - 🔴 **Red** = pushes price **down**
        """)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
        ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient (effect on Price per 1-unit increase)")
        ax.set_title("Linear Regression — Feature Coefficients")
        plt.tight_layout()
        st.pyplot(fig)
 
        st.markdown("---")
        st.subheader("📋 Coefficient Table")
        st.dataframe(coef_df.reset_index(drop=True), use_container_width=True)
        st.caption(f"Model intercept (baseline price): ${model_exp.intercept_:,.2f}")
 
    else:
        if model_choice == "Decision Tree":
            model_exp = DecisionTreeRegressor(max_depth=5, random_state=42)
        else:
            model_exp = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
 
        model_exp.fit(X_train_exp, y_train_exp)
 
        importance_df = pd.DataFrame({
            "Feature": features_exp,
            "Importance": model_exp.feature_importances_
        }).sort_values("Importance", ascending=False)
 
        st.subheader(f"📊 Feature Importance — {model_choice}")
        st.markdown("""
        Each bar shows what percentage of the model's predictions are driven by that feature.
        A longer bar = that feature explains more of the variation in house price.
        """)
 
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color="#3498db")
        ax.set_xlabel("Importance Score (higher = more influential)")
        ax.set_title(f"{model_choice} — Feature Importance")
        plt.tight_layout()
        st.pyplot(fig)
 
        st.markdown("---")
        st.subheader("📋 Importance Table")
        st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)
 
        top_feature = importance_df.iloc[0]["Feature"]
        top_score = importance_df.iloc[0]["Importance"]
        st.info(f"🏆 Most influential feature: **{top_feature}** — accounts for {top_score*100:.1f}% of prediction power")

# --- hyperparamter tuning Page ---
elif page == "Hyperparameter Tuning 📈":
    st.header("05 Hyperparameter Tuning 📈")
    st.markdown("""
    ### 🎯 What is Hyperparameter Tuning?
    A **hyperparameter** is a setting you choose before training a model — like how deep
    a decision tree is allowed to grow. Tuning means trying different values and tracking
    which combination gives the best performance.
 
    **How to use this page:**
    1. Pick a model and adjust its settings using the sliders
    2. Click **Run Experiment** to train and record the result
    3. Repeat with different settings to compare
    4. The table below tracks all your experiments so you can find the best one
    """)
    st.markdown("---")
 
    df_tune = df.copy()
    df_tune['was_renovated'] = (df_tune['Renovation Year'] != 0).astype(int)
    df_tune = df_tune.drop(columns=['Renovation Year'])
 
    features_tune = df_tune.columns.tolist()
    features_tune.remove("Price")
 
    X_tune = df_tune[features_tune]
    y_tune = df_tune["Price"]
 
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_tune, y_tune, test_size=0.2, random_state=42
    )
 
    if "experiment_log" not in st.session_state:
        st.session_state.experiment_log = []
 
    st.subheader("⚙️ Configure Your Experiment")
 
    col1, col2 = st.columns(2)
 
    with col1:
        tune_model_name = st.selectbox(
            "Select Model",
            ["Linear Regression", "Decision Tree", "Random Forest"],
            key="tune_model"
        )
 
    with col2:
        tune_params = {}
        if tune_model_name == "Decision Tree":
            tune_params['max_depth'] = st.slider(
                "Max Depth",
                min_value=1, max_value=20, value=5,
                help="How deep the tree can grow. Deeper = more complex. Too deep = overfitting."
            )
        elif tune_model_name == "Random Forest":
            tune_params['n_estimators'] = st.slider(
                "Number of Trees",
                min_value=10, max_value=300, value=100, step=10,
                help="How many trees in the forest. More trees = more accurate but slower."
            )
            tune_params['max_depth'] = st.slider(
                "Max Depth",
                min_value=1, max_value=20, value=5,
                help="Max depth of each tree in the forest."
            )
        else:
            st.info("Linear Regression has no hyperparameters to tune.")
 
    if st.button("🚀 Run Experiment"):
 
        if tune_model_name == "Linear Regression":
            tune_model = LinearRegression()
        elif tune_model_name == "Decision Tree":
            tune_model = DecisionTreeRegressor(**tune_params, random_state=42)
        elif tune_model_name == "Random Forest":
            tune_model = RandomForestRegressor(**tune_params, random_state=42)

        tune_model.fit(X_train_t, y_train_t)
        tune_preds = tune_model.predict(X_test_t)
 
        tune_mae = metrics.mean_absolute_error(y_test_t, tune_preds)
        tune_mse = metrics.mean_squared_error(y_test_t, tune_preds)
        tune_r2  = metrics.r2_score(y_test_t, tune_preds)
 
        log_entry = {
            "Model": tune_model_name,
            "MAE": round(tune_mae, 2),
            "MSE": round(tune_mse, 2),
            "R²": round(tune_r2, 4),
        }
        log_entry.update(tune_params)
 

        st.session_state.experiment_log.append(log_entry)
        st.success(f"✅ Experiment recorded! R² = {tune_r2:.4f}")
 
  
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"${tune_mae:,.2f}")
        c2.metric("MSE", f"${tune_mse:,.2f}")
        c3.metric("R²", f"{tune_r2:.4f}")
 
    st.markdown("---")
 
    st.subheader("📋 Experiment Log")
    st.markdown("Every experiment you run gets recorded here. Compare to find the best settings.")
 
    if st.session_state.experiment_log:
        log_df = pd.DataFrame(st.session_state.experiment_log)
        st.dataframe(log_df, use_container_width=True)

        best_idx = log_df["R²"].idxmax()
        best_row = log_df.loc[best_idx]
        st.success(
            f"🏆 Best experiment so far: **{best_row['Model']}** "
            f"with R² = **{best_row['R²']}** and MAE = **${best_row['MAE']:,.2f}**"
        )
 
        if st.button("🗑️ Clear Experiment Log"):
            st.session_state.experiment_log = []
            st.info("Log cleared. Start new experiments above.")
 
    else:
        st.info("No experiments recorded yet. Configure settings above and click Run Experiment.")

# --- conclusion ---
elif page == "Conclusion 📖":
    st.header("06 Conclusion 📖")
    st.markdown("""
    ### 🏠 Project Summary
    This application set out to solve a real problem in the Indian real estate market:
    **predicting house prices accurately** to help buyers, sellers, and agents make
    better-informed decisions without relying on guesswork.
 
    We used a dataset of **14,619 houses** across India with features including:
    - Living area, number of bedrooms, bathrooms, and floors
    - Condition and grade of the house
    - Renovation status
    - Proximity to schools and distance from airports
    """)
 
    st.markdown("---")
 
    st.subheader("📊 What the Models Told Us")
    st.markdown("""
    We trained and compared three models:
 
    - **Random Forest** — highest R² score (~0.82), explaining the most variation in house prices
    - **Decision Tree** — moderate performance (R² averaging 0.6–0.72)
    - **Linear Regression** — lagged behind both, since its straight-line assumption can't
      capture the more complex, non-linear relationships between features like living area and price
    """)
 
    st.markdown("---")
 
    st.subheader("🏆 Key Findings — What Drives House Price?")
    st.markdown("""
    On our Explainability page, the feature importance rankings were:
 
    - **Grade of the house** and **living area** — the two most important features in both
      Decision Tree and Random Forest, with the longest bars by a wide margin
    - **Postal code** and **built year** — the third and fourth most important features,
      though noticeably less significant than the top two
    - This confirms that the **size and quality** of a home are the strongest, most
      reliable predictors of price across the entire dataset
 
    A major difference appeared in the Linear Regression chart, however — it ranked
    **waterfront present** as by far the single strongest driver of price, with a
    coefficient far larger than every other feature combined. This does **not** actually
    mean waterfront access is the most important factor in reality.
    """)
 
    st.markdown("---")
 
    st.subheader("🔍 Why the Models Disagreed")
    st.markdown("""
    - **Linear Regression's coefficient** measures price change per single 1-unit increase.
      Waterfront is a binary feature (0 or 1), and the few waterfront homes in the dataset
      sold for dramatically more — so that 0-to-1 jump produces an inflated coefficient
    - **Decision Tree and Random Forest** measure importance by how often a feature
      actually helps split houses into accurate price groups across the *entire* dataset
    - Since nearly every house has a different living area and grade, those features get
      used constantly and earn high importance — waterfront, being rare, barely registers
    - Because Random Forest had the highest overall R², we trust its feature ranking the
      most: **living area and grade are the genuine, broadly applicable drivers of price**,
      not waterfront access
    """)
 
    st.markdown("---")
 
    st.subheader("💼 Business Value")
    st.markdown("""
    The prediction page of this app gives real estate professionals a tool to:
 
    - **Estimate a home's price** before listing it, using the features that matter most
    - **Prioritize renovations or upgrades** that genuinely add value — like increasing
      living area or improving the home's overall grade — rather than chasing rare
      features that may not generalize
    - **Compare locations** using postal code trends as a secondary signal
    - **Make faster decisions** backed by data rather than intuition alone
    """)
 
    st.markdown("---")
 
 
    st.subheader("✅ Final Takeaway")
    st.success(
        "Random Forest was the clear winner, both in raw predictive accuracy and in giving "
        "a feature importance ranking that reflects genuine, dataset-wide patterns. Living area "
        "and grade of the house are the strongest, most trustworthy predictors of price — "
        "providing a reliable foundation for real estate decision-making."
    )
    
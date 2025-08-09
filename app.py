# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import seaborn as sns


try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

# ---------------------------
# Load Data & Model
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv('D:/3 year 2sem/Machine_Learning_Model_Deployment_Streamlit__Boston_Housing_Dataset/Data/BostonHousing.csv')

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, feature_names

df = load_data()
model, feature_names = load_model()

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Data Exploration", "Visualisations", "Model Prediction", "Model Performance"]
)

# ---------------------------
# Project Overview
# ---------------------------
if section == "Project Overview":
    st.title("🏠 Boston Housing Price Prediction")
    st.markdown("""
    This app predicts **median house prices** in Boston suburbs based on multiple housing attributes.
    
    **Dataset:** BostonHousing.csv  
    **Model:** Best-performing regression model selected from Linear Regression & Random Forest Regressor.  
    """)

# ---------------------------
# Data Exploration
# ---------------------------
elif section == "Data Exploration":
    st.title("📊 Data Exploration")
    st.write("### Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Data Types:")
    st.write(df.dtypes)

    st.write("### Sample Data")
    st.dataframe(df.head())

    st.write("### Interactive Filtering")
    col_to_filter = st.selectbox("Select column to filter", df.columns)
    unique_vals = df[col_to_filter].unique()
    selected_val = st.selectbox("Select value", unique_vals)
    filtered_df = df[df[col_to_filter] == selected_val]
    st.write(f"Filtered Data (where {col_to_filter} = {selected_val}):")
    st.dataframe(filtered_df)

# ---------------------------
# Visualisations
# ---------------------------
elif section == "Visualisations":
    st.title("📈 Visualisations")

    # 1. Distribution of Target
    st.subheader("Distribution of House Prices (medv)")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["medv"], kde=True, ax=ax1)
    st.pyplot(fig1)

    # 2. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # 3. Scatter plot with user selection
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("X-axis", df.columns, index=5)
    y_axis = st.selectbox("Y-axis", df.columns, index=len(df.columns)-1)
    fig3, ax3 = plt.subplots()
    ax3.scatter(df[x_axis], df[y_axis], alpha=0.6)
    ax3.set_xlabel(x_axis)
    ax3.set_ylabel(y_axis)
    st.pyplot(fig3)

# ---------------------------
# Model Prediction
# ---------------------------
elif section == "Model Prediction":
    st.title("🤖 Model Prediction")
    st.write("Enter feature values to predict median home price:")

    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(
            f"{feature}",
            value=float(df[feature].mean())
        )

    if st.button("Predict"):
        input_array = np.array(list(user_input.values())).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        st.success(f"Predicted Median Home Value: ${prediction * 1000:,.2f}")

# ---------------------------
# Model Performance
# ---------------------------
elif section == "Model Performance":
    st.title("📊 Model Performance")

    # Load test data for metrics
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    X = df[feature_names]
    y = df["medv"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)

    # Evaluation metrics
    st.write("### Evaluation Metrics")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

    # Residual Plot
    st.write("### Residual Plot")
    fig4, ax4 = plt.subplots()

    try:
        import statsmodels  # Check if statsmodels is installed
        sns.residplot(
            x=y_pred,
            y=y_test - y_pred,
            lowess=True,
            ax=ax4,
            line_kws={"color": "red"}
        )
    except ImportError:
        st.warning("⚠ statsmodels not found. LOWESS smoothing disabled.")
        sns.residplot(
            x=y_pred,
            y=y_test - y_pred,
            ax=ax4,
            line_kws={"color": "red"}
        )

    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Residuals")
    st.pyplot(fig4)

    # Model comparison results
    st.write("### Model Comparison Results")
    st.write("The model selected was the one with the highest cross-validation R² score during training.")
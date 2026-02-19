import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from imblearn.pipeline import Pipeline
import joblib
import os

# Set page config
st.set_page_config(page_title="Calories Burnt Prediction", layout="wide")

st.title("🔥 Calories Burnt Prediction Model")
st.markdown("Predict the calories burnt during exercise using ML models")

# Sidebar for navigation
page = st.sidebar.radio("Select Page", ["Home", "Train Model", "Make Prediction", "Model Performance"])

# Function to load data
@st.cache_data
def load_data():
    try:
        exercise = pd.read_csv(r"C:\Users\T.B\Downloads\exercise.csv")
        calories = pd.read_csv(r"C:\Users\T.B\Downloads\calories.csv")
        df = exercise.merge(calories, on="User_ID")
        df = df.drop("User_ID", axis=1)
        df["Gender"] = df["Gender"].map({"male": 0, "female": 1})
        return df
    except FileNotFoundError:
        return None

# Function to train all models
@st.cache_resource
def train_models(df):
    X = df.drop("Calories", axis=1)
    y = df["Calories"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), X.columns)
    ])
    
    model_grids = {
        "Linear Regression": {
            "model": LinearRegression(),
            "params": {}
        },
        "Ridge": {
            "model": Ridge(),
            "params": {"model__alpha": [0.01, 0.1, 1, 10, 100]}
        },
        "Lasso": {
            "model": Lasso(max_iter=10000),
            "params": {"model__alpha": [0.001, 0.01, 0.1, 1]}
        },
        "SVR": {
            "model": SVR(),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest": {
            "model": RandomForestRegressor(),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20]
            }
        },
        "XGBoost": {
            "model": XGBRegressor(),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__max_depth": [3, 6, 10]
            }
        }
    }
    
    results = {}
    best_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, config) in enumerate(model_grids.items()):
        status_text.text(f"Training {name}...")
        
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", config["model"])
        ])
        
        grid = GridSearchCV(
            pipeline,
            config["params"],
            cv=5,
            scoring="r2",
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)
        
        results[name] = {
            "Best R2": r2_score(y_test, preds),
            "MSE": mean_squared_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "Best Params": grid.best_params_
        }
        
        best_models[name] = best_model
        progress_bar.progress((idx + 1) / len(model_grids))
    
    status_text.empty()
    progress_bar.empty()
    
    results_df = pd.DataFrame(results).T
    best_model_name = results_df["Best R2"].idxmax()
    final_model = best_models[best_model_name]
    
    # Save model
    joblib.dump(final_model, "best_model.pkl")
    
    return results_df, best_models, final_model, best_model_name, X_test, y_test, X

# Home Page
if page == "Home":
    st.header("Welcome to Calories Burnt Prediction")
    
    df = load_data()
    if df is not None:
        st.success("✅ Data loaded successfully!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", df.shape[1] - 1)
        with col3:
            st.metric("Average Calories", f"{df['Calories'].mean():.2f}")
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        st.subheader("Data Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].hist(df["Calories"], bins=30, edgecolor='black', color='skyblue')
        axes[0].set_title("Calories Distribution")
        axes[0].set_xlabel("Calories")
        axes[0].set_ylabel("Frequency")
        
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=axes[1], fmt=".2f")
        axes[1].set_title("Correlation Matrix")
        
        st.pyplot(fig)
    else:
        st.error("❌ Could not load data. Please check the file paths.")

# Train Model Page
elif page == "Train Model":
    st.header("Train Machine Learning Models")
    
    df = load_data()
    if df is not None:
        if st.button("🚀 Start Training", key="train_btn"):
            results_df, best_models, final_model, best_model_name, X_test, y_test, X = train_models(df)
            
            st.success("✅ Training completed!")
            
            st.subheader("Model Performance Comparison")
            results_sorted = results_df.sort_values("Best R2", ascending=False)
            st.dataframe(results_sorted)
            
            st.subheader(f"🏆 Best Model: {best_model_name}")
            st.metric("R² Score", f"{results_df.loc[best_model_name, 'Best R2']:.4f}")
            
            # Visualize comparison
            fig, ax = plt.subplots(figsize=(10, 5))
            results_df["Best R2"].sort_values(ascending=False).plot(kind="bar", ax=ax, color='steelblue')
            ax.set_title("Model Comparison - R² Scores")
            ax.set_ylabel("R² Score")
            ax.set_xlabel("Model")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.error("❌ Could not load data.")

# Make Prediction Page
elif page == "Make Prediction":
    st.header("Predict Calories Burnt")
    
    # Check if model exists
    if os.path.exists("best_model.pkl"):
        model = joblib.load("best_model.pkl")
        
        st.subheader("Enter Exercise Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            age = st.slider("Age (years)", 18, 80, 25)
            height = st.slider("Height (cm)", 140, 220, 175)
            weight = st.slider("Weight (kg)", 40, 150, 70)
        
        with col2:
            duration = st.slider("Duration (minutes)", 5, 120, 20)
            heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 100)
            body_temp = st.slider("Body Temperature (°C)", 36.0, 41.0, 39.5)
        
        # Prepare data for prediction
        gender_encoded = 0 if gender == "Male" else 1
        
        new_data = pd.DataFrame({
            "Gender": [gender_encoded],
            "Age": [age],
            "Height": [height],
            "Weight": [weight],
            "Duration": [duration],
            "Heart_Rate": [heart_rate],
            "Body_Temp": [body_temp]
        })
        
        if st.button("🔮 Predict Calories"):
            prediction = model.predict(new_data)[0]
            
            st.success(f"### Predicted Calories Burnt: **{prediction:.2f} kcal**")
            
            # Additional insights
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📊 Duration: {duration} minutes")
            with col2:
                st.info(f"💓 Heart Rate: {heart_rate} bpm")
    else:
        st.warning("⚠️ Model not found. Please train the model first on the 'Train Model' page.")

# Model Performance Page
elif page == "Model Performance":
    st.header("Detailed Model Performance")
    
    df = load_data()
    if df is not None and os.path.exists("best_model.pkl"):
        model = joblib.load("best_model.pkl")
        X = df.drop("Calories", axis=1)
        y = df["Calories"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        preds = model.predict(X_test)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
        with col2:
            st.metric("MSE", f"{mean_squared_error(y_test, preds):.2f}")
        with col3:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
        
        st.subheader("Actual vs Predicted")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].scatter(y_test, preds, alpha=0.6)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel("Actual Calories")
        axes[0].set_ylabel("Predicted Calories")
        axes[0].set_title("Actual vs Predicted")
        
        residuals = y_test - preds
        axes[1].scatter(preds, residuals, alpha=0.6)
        axes[1].axhline(y=0, color="red", linestyle="--")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Residual Plot")
        
        st.pyplot(fig)
        
        # Cross validation
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        st.subheader("Cross-Validation Scores")
        st.write(f"Scores: {scores}")
        st.write(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
    else:
        st.warning("⚠️ Please train the model first.")

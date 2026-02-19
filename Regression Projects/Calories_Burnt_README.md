# Calories Burnt Prediction

A machine learning Streamlit application that predicts calories burnt during exercise using various regression models.

## Features

- **Multiple ML Models**: Linear Regression, Ridge, Lasso, SVR, Decision Tree, Random Forest, and XGBoost
- **Model Comparison**: Compare performance metrics (R², MSE, RMSE) across all models
- **Interactive Predictions**: Use sliders to input exercise parameters and get real-time predictions
- **Data Visualization**: Distribution plots, correlation matrix, and performance analysis
- **Cross-validation**: Detailed model evaluation with cross-validation scores

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abdallah-mo/Ai-projects.git
cd "Regression Projects"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Pages

- **Home**: Data overview and exploratory analysis
- **Train Model**: Train all models and compare performance
- **Make Prediction**: Interactive tool to predict calories for custom inputs
- **Model Performance**: Detailed metrics and evaluation plots

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- streamlit
- imbalanced-learn
- joblib

## Data Files

The app expects two CSV files:
- `exercise.csv` - Exercise features
- `calories.csv` - Calorie targets

## License

MIT License

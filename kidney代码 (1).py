#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- Your Provided Code (Adapted for Streamlit) ---

# Load data
df = pd.read_excel('5.19交集特征.xlsx')

# Define variables
continuous_vars = [
    '入选时FiO2', 'Lym_D1', 'Hb(g/L)_D2', 'BUN(mmolL)_D2', 'Cl(mmolL)_D1',
    'PT(s)_D2', 'PTA(%)_D2', 'Fib(gL)_D2', 'PO2/FiO2(mmHg)_D2', 'HCO3_D2',
    'Change of white blood cell count', '48-hour fluid balance',
    'APACHE Ⅱ score_at the time of inclusion', 'CTnI(ngml)_D2', 'BUN(mmolL)_D1',
    'DBIL(μmolL)_D2'
]
categorical_vars = [
    'Predisposing factors for ARDS', 'Chronic lung disease', 'Respiratory support_D2'
]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_vars),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_vars)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(df)

# Get feature names
try:
    feature_names = (
        continuous_vars +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_vars))
    )
except AttributeError:
    feature_names = (
        continuous_vars +
        list(preprocessor.named_transformers_['cat'].get_feature_names(categorical_vars))
    )

X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
X = X_processed_df
y = df['急性肾衰竭']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)


# --- Streamlit App Interface ---

st.title("Acute Kidney Failure Prediction")

# --- 1. User Input for X values ---
st.header("1. Enter Patient Data")

user_input = {}

st.subheader("Continuous Variables")
# Create input fields for continuous variables
cont_cols = st.columns(2)
for i, var in enumerate(continuous_vars):
    with cont_cols[i % 2]:
        # Default value could be mean or median from training data, here using 0.0 for simplicity
        default_val = X[var].mean() if var in X.columns else 0.0
        user_input[var] = st.number_input(f"{var}", value=float(default_val), format="%.4f", step=0.1)

st.subheader("Categorical Variables")
# Create input fields for categorical variables
# We need to know the categories from the training data's encoder
fitted_encoder = preprocessor.named_transformers_['cat']
try:
    categories = fitted_encoder.categories_
except AttributeError:
    # Fallback if categories_ is not directly available
    categories = [np.unique(df[col].astype(str)) for col in categorical_vars]

cat_options_dict = dict(zip(categorical_vars, categories))

for var in categorical_vars:
    options = cat_options_dict.get(var, ['Unknown'])
    # Default to the first category or a placeholder
    default_option = options[0] if len(options) > 0 else 'Unknown'
    user_input[var] = st.selectbox(f"{var}", options=options, index=0)

# --- 2. Model Parameter Adjustment (for Training) ---
st.header("2. Set Model Parameters for Training")

# Using sidebar for parameters
with st.sidebar:
    st.subheader("SVM Parameters")
    # Default parameters matching your original code snippet
    selected_kernel = st.selectbox("Kernel", options=['linear', 'rbf', 'poly'], index=0) # Default 'linear'
    selected_C = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01) # Default 1.0
    selected_class_weight = st.selectbox("Class Weight", options=[None, 'balanced'], index=1) # Default 'balanced'

# --- 3. Prediction Button and Logic ---
if st.button("Train Model and Predict"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([user_input])

    # --- Train the model with selected parameters ---
    try:
        # Use the train/test split defined earlier
        svc = SVC(
            kernel=selected_kernel,
            C=selected_C,
            class_weight=selected_class_weight,
            probability=True, # Required for predict_proba
            random_state=999
        )
        svc.fit(X_train, y_train) # Train on the training set
        st.success("Model trained successfully with selected parameters!")

        # Apply the same preprocessing pipeline to input data
        input_processed = preprocessor.transform(input_data)

        # Make prediction using the newly trained model
        # Predicted class
        prediction = svc.predict(input_processed)[0]
        # Prediction probabilities
        prediction_proba = svc.predict_proba(input_processed)[0]

        # Display results
        st.header("Prediction Result")
        # st.write(f"**Predicted Class:** {prediction}")
        st.metric(label="Predicted Probability of Acute Kidney Failure", value=f"{prediction_proba[1]:.4f}")

    except Exception as e:
        st.error(f"An error occurred during model training or prediction: {e}")


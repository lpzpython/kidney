import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- Set wide layout ---
st.set_page_config(layout="wide")

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
# Combine all variables for unified input
all_vars = continuous_vars + categorical_vars

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

# --- 1. User Input for X values (Unified, 4 columns) ---
st.header("1. Enter Patient Data")

user_input = {}

# Create input fields for all variables in 4 columns
# Combine continuous and categorical for unified handling in layout
input_cols = st.columns(4) # Changed to 4 columns
for i, var in enumerate(all_vars):
    with input_cols[i % 4]: # Cycle through 4 columns
        if var in continuous_vars:
            # Handle continuous variables
            default_val = X[var].mean() if var in X.columns else 0.0
            user_input[var] = st.number_input(f"{var}", value=float(default_val), format="%.4f", step=0.1)
        else: # Handle categorical variables
            # Get categories for categorical variables
            fitted_encoder = preprocessor.named_transformers_['cat']
            try:
                # Find the index of the categorical variable in the transformer's input list
                cat_var_index = categorical_vars.index(var)
                options = fitted_encoder.categories_[cat_var_index]
            except (AttributeError, ValueError, IndexError):
                # Fallback if categories_ is not directly available or index error
                options = np.unique(df[var].astype(str))
            # Default to the first category or a placeholder
            default_option = options[0] if len(options) > 0 else 'Unknown'
            user_input[var] = st.selectbox(f"{var}", options=options, index=0)

# --- 2. Model Parameter Adjustment (Moved below X input, no sidebar) ---
st.header("2. Set Model Parameters for Training")

# Create columns for parameters to keep them organized
param_cols = st.columns(3)
with param_cols[0]:
    selected_kernel = st.selectbox("Kernel", options=['linear', 'rbf', 'poly'], index=0) # Default 'linear'
with param_cols[1]:
    selected_C = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01) # Default 1.0
with param_cols[2]:
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





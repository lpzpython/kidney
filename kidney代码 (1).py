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
# Note: Ensure '5.19交集特征.xlsx' is in the same directory or provide the full path
try:
    df = pd.read_excel('5.19交集特征.xlsx')
except FileNotFoundError:
    st.error("Error, file not found")
    st.stop()

df.rename(columns={"入选时FiO2":"FiO2",
                   "APACHE Ⅱ score_at the time of inclusion":"APACHE II",
                  "Lym_D1":"Lym(*10^9/L)_D1",
                  "BUN(mmolL)_D2":"BUN(mmol/L)_D2",
                  "Cl(mmolL)_D1":"Cl(mmol/L)_D1",
                  "Fib(gL)_D2":"Fib(g/L)_D2",
                  "HCO3_D2":"HCO3(mmol/L)_D2",
                  "Change of white blood cell count":"Change of white blood cell count(*10^9/L)",
                  "48-hour fluid balance":"48-hour fluid balnce(ml)",
                  "CTnI(ngml)_D2":"CTnl(ng/ml)_D2",
                  "BUN(mmolL)_D1":"BUN(mmol/L)_D1",
                  "DBIL(μmolL)_D2":"DBIL(μmol/L)D2"},inplace=True)

# Define variables
continuous_vars = [
    'FiO2', 'Lym(*10^9/L)_D1', 'Hb(g/L)_D2', 'BUN(mmol/L)_D2', 'Cl(mmol/L)_D1',
    'PT(s)_D2', 'PTA(%)_D2', 'Fib(g/L)_D2', 'PO2/FiO2(mmHg)_D2', 'HCO3(mmol/L)_D2',
    'Change of white blood cell count(*10^9/L)', '48-hour fluid balnce(ml)',
    'APACHE II', 'CTnl(ng/ml)_D2', 'BUN(mmol/L)_D1',
    'DBIL(μmol/L)D2'
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

# Centered Title
st.markdown("<h1 style='text-align: center;'>Support Vector Machine model for predicting ARDS patients with Late acute AKI</h1>", unsafe_allow_html=True)

# --- 1. User Input for X values (Unified, 4 columns) ---
st.header("1. Enter Patient Data")

user_input = {} # summarize user input data
input_valid = True # Flag to check if all inputs are valid
# Create input fields for all variables in 4 columns
# Combine continuous and categorical for unified handling in layout
input_cols = st.columns(4) # Changed to 4 columns
for i, var in enumerate(all_vars):
    with input_cols[i % 4]: # Cycle through 4 columns
        if var in continuous_vars:
            # Handle continuous variables - No default value
            if var =="FiO2":
                user_val = st.number_input(f"{var}",value=None,format="%.4f", step=0.01, placeholder="please enter,e.g.,0.6")
            else:
                user_val = st.number_input(f"{var}", value=None, format="%.4f",step=0.01, placeholder="please enter")
            if user_val is None:
                input_valid = False
                #st.warning(f"请输入 {var} 的值")
            user_input[var] = user_val
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
            # UI - No default selection, user must choose
            selected_option = st.selectbox(f"{var}", options=options, index=None, placeholder="please enter")
            if selected_option is None:
                input_valid = False
                #st.warning(f"请选择 {var} 的值")
            user_input[var] = selected_option

# --- 2. Model Parameter Display (Fixed, no user selection) ---
st.header("2. Model Parameters (Fixed)")

# Display fixed parameters
# Store fixed parameters
FIXED_KERNEL = 'linear'
FIXED_C = 1.0
FIXED_CLASS_WEIGHT = 'balanced'

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Kernel", value=FIXED_KERNEL)
with col2:
    st.metric(label="Regularization Parameter (C)", value=FIXED_C )
with col3:
    st.metric(label="Class Weight", value=FIXED_CLASS_WEIGHT)

# --- 3. Prediction Button and Logic ---
if st.button("Train Model and Predict"):
    if not input_valid:
        st.error("error, please check all X is inputed")
    else:
        # Create a DataFrame from user input
        input_data = pd.DataFrame([user_input])

        # --- Train the model with fixed parameters ---
        try:
            # Use the train/test split defined earlier
            # model
            svc = SVC(
                kernel=FIXED_KERNEL,
                C=FIXED_C,
                class_weight=FIXED_CLASS_WEIGHT,
                probability=True, # Required for predict_proba
                random_state=999
            )
            svc.fit(X_train, y_train) # Train on the training set
            st.success("Model trained successfully with fixed parameters!")

            # Apply the same preprocessing pipeline to input data
            input_processed = preprocessor.transform(input_data)

            # Make prediction using the newly trained model
            # Prediction probabilities
            prediction_proba = svc.predict_proba(input_processed)[0]
            
            # Display results
            st.header("Prediction Result")
            # Assuming class 1 is 'Acute Kidney Injury'
            prob_label = "Predicted Probability of Acute Kidney Injury"
            st.metric(label=prob_label, value=f"{prediction_proba[1]*100:.2f}%") # Displaying probability of class 1

        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {e}")


# --- Disclaimer Section at the Bottom ---
st.markdown("---") # Horizontal line separator
disclaimer_text = """
**Disclaimer:**

Supplement:
*   D1 and D2 represent the first day and the second day after ARDS diagnosis, respectively.
*   APACHE II and FIO₂ were recorded on the first day after ARDS diagnosis.
*   Change of white blood cell count was calculated as the difference between the count on D2 and D1.
*   48-hour fluid balance represents the total intake and output volume during the 2 days after ARDS diagnosis.
*   Respiratory support_D2_1 = oxygen therapy.
*   Respiratory support_D2_2 = non-invasive ventilation.
*   Respiratory support_D2_3 = invasive mechanical ventilation.
*   Predisposing factors for ARDS_1 = pneumonia.
*   Predisposing factors for ARDS_0 = other factors.
*   Chronic lung disease_1 = with Chronic lung disease.
*   Chronic lung disease_0 = without Chronic lung disease.
"""
st.markdown(disclaimer_text)





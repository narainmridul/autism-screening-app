import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load('random_forest_model.pkl')
training_columns = pd.read_csv('training_columns.csv', header=None).squeeze()

def preprocess_data(data, training_columns):
    # Handle missing values
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Create dummy variables for categorical columns
    data = pd.get_dummies(data, columns=non_numeric_cols, dtype=int)
    
    # Debug: Check columns after get_dummies
    # st.write("Columns after get_dummies:", data.columns.tolist())
    
    # Align columns with training_columns
    missing_cols = set(training_columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0  # Add missing columns with zeros
    extra_cols = set(data.columns) - set(training_columns)
    data = data.drop(columns=extra_cols, errors='ignore')  # Drop extra columns
    
    # Reorder columns to match training_columns
    data = data[training_columns]
    
    # Debug: Verify final columns
    # st.write("Final processed columns:", data.columns.tolist())
    
    return data

st.title("Autism Screening Prediction")
st.write("Enter patient details to predict ASD.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        a1 = st.number_input("A1_Score (0/1)", min_value=0, max_value=1, value=0)
        a2 = st.number_input("A2_Score (0/1)", min_value=0, max_value=1, value=0)
        a3 = st.number_input("A3_Score (0/1)", min_value=0, max_value=1, value=0)
        a4 = st.number_input("A4_Score (0/1)", min_value=0, max_value=1, value=0)
        a5 = st.number_input("A5_Score (0/1)", min_value=0, max_value=1, value=0)
    with col2:
        a6 = st.number_input("A6_Score (0/1)", min_value=0, max_value=1, value=0)
        a7 = st.number_input("A7_Score (0/1)", min_value=0, max_value=1, value=0)
        a8 = st.number_input("A8_Score (0/1)", min_value=0, max_value=1, value=0)
        a9 = st.number_input("A9_Score (0/1)", min_value=0, max_value=1, value=0)
        a10 = st.number_input("A10_Score (0/1)", min_value=0, max_value=1, value=0)
    
    age = st.number_input("Age", min_value=0.0, value=30.0)
    gender = st.selectbox("Gender", ['m', 'f'])
    ethnicity = st.text_input("Ethnicity", value="White-European")
    jaundice = st.selectbox("Jaundice", ['no', 'yes'])
    austim = st.selectbox("Family Autism", ['no', 'yes'])
    country = st.text_input("Country", value="United States")
    app_before = st.selectbox("Used App Before", ['no', 'yes'])
    age_desc = st.text_input("Age Description", value="18 and more")
    relation = st.text_input("Relation", value="Self")

    submit = st.form_submit_button("Predict")

if submit:
    patient_data = pd.DataFrame({
        'A1_Score': [a1], 'A2_Score': [a2], 'A3_Score': [a3], 'A4_Score': [a4], 'A5_Score': [a5],
        'A6_Score': [a6], 'A7_Score': [a7], 'A8_Score': [a8], 'A9_Score': [a9], 'A10_Score': [a10],
        'age': [age], 'gender': [gender], 'ethnicity': [ethnicity], 'jundice': [jaundice],
        'austim': [austim], 'contry_of_res': [country], 'used_app_before': [app_before],
        'age_desc': [age_desc], 'relation': [relation]
    })
    
    processed_data = preprocess_data(patient_data, training_columns)
    prediction = model.predict(processed_data)[0]
    result = 'YES (ASD)' if prediction == 1 else 'NO (No ASD)'
    st.success(f"Prediction: {result}")
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Autism Screening", layout="centered")

# Cache model and columns
@st.cache_data
def load_data():
    model = joblib.load('random_forest_model.pkl')
    columns = pd.read_csv('training_columns.csv', header=None).squeeze()
    return model, columns

model, training_columns = load_data()

def preprocess_data(data, training_columns):
    # Copy to avoid modifying input
    data = data.copy()
    
    # Handle missing values
    data = data.fillna({
        col: data[col].median() if data[col].dtype in ['int64', 'float64'] else data[col].mode()[0]
        for col in data.columns
    })
    
    # Create dummy variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, dtype=int)
    
    # Align with training columns
    missing_cols = set(training_columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    data = data[training_columns]
    
    return data

st.header("Autism Screening Tool")
st.markdown("""
    This tool predicts Autism Spectrum Disorder (ASD) based on patient data.  
    - **A1-A10 Scores**: Enter 0 (No) or 1 (Yes) for behavioral questions.
    - **Other Fields**: Provide accurate details (e.g., age, gender).
    - All fields are required. Click **Predict** to see the result.
""")
st.markdown("<style>.stForm {border: 1px solid #ddd; padding: 20px; border-radius: 10px;}</style>", unsafe_allow_html=True)

st.subheader("Behavioral Scores (0 = No, 1 = Yes)")
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
    
    st.subheader("Patient Details")
    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("Age", min_value=1.0, max_value=100.0, value=30.0)
        gender = st.selectbox("Gender", ['Male', 'Female'], format_func=lambda x: x)
        ethnicity = st.selectbox("Ethnicity", [
            "White-European", "Asian", "Black", "Hispanic", "Latino", 
            "Middle Eastern", "Others", "Pasifika", "South Asian", "Turkish"
        ])
        jaundice = st.selectbox("Born with Jaundice", ['No', 'Yes'])
        austim = st.selectbox("Family History of Autism", ['No', 'Yes'])
    with col4:
        country = st.selectbox("Country of Residence", [
            "United States", "United Kingdom", "India", "Brazil", "Australia",
            "Canada", "New Zealand", "Other"
        ])
        app_before = st.selectbox("Previously Used App", ['No', 'Yes'])
        age_desc = st.text_input("Age Group", value="18 and more")
        relation = st.text_input("Relation to Patient", value="Self")

    submit = st.form_submit_button("Predict")
    
if submit:
    # Validate inputs
    required_fields = {
        'Ethnicity': ethnicity, 'Country': country, 
        'Age Description': age_desc, 'Relation': relation
    }
    empty_fields = [k for k, v in required_fields.items() if not v]
    if empty_fields:
        st.error(f"Please fill in: {', '.join(empty_fields)}")
    elif age <= 0:
        st.error("Age must be greater than 0.")
    else:
        patient_data = pd.DataFrame({
            'A1_Score': [a1], 'A2_Score': [a2], 'A3_Score': [a3], 'A4_Score': [a4], 'A5_Score': [a5],
            'A6_Score': [a6], 'A7_Score': [a7], 'A8_Score': [a8], 'A9_Score': [a9], 'A10_Score': [a10],
            'age': [age], 'gender': [gender], 'ethnicity': [ethnicity], 'jundice': [jaundice],
            'austim': [austim], 'contry_of_res': [country], 'used_app_before': [app_before],
            'age_desc': [age_desc], 'relation': [relation]
        })
        
        try:
            processed_data = preprocess_data(patient_data, training_columns)
            prediction = model.predict(processed_data)[0]
            result = 'YES (ASD)' if prediction == 1 else 'NO (No ASD)'
            st.success(f"Prediction: {result}")
            if prediction == 1:
                st.markdown("**Note**: A 'YES (ASD)' prediction suggests potential ASD traits. Consult a healthcare professional.")
            else:
                st.markdown("**Note**: A 'NO (No ASD)' prediction indicates lower likelihood of ASD.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
import streamlit as st
import pandas as pd
import joblib

# Set page config first
st.set_page_config(page_title="Autism Screening", layout="centered")

# Cache model and columns
@st.cache_data
def load_data():
    model = joblib.load('random_forest_model.pkl')
    columns = pd.read_csv('training_columns.csv', header=None).squeeze()
    return model, columns

model, training_columns = load_data()

def preprocess_data(data, training_columns):
    data = data.copy()
    data = data.fillna({
        col: data[col].median() if data[col].dtype in ['int64', 'float64'] else data[col].mode()[0]
        for col in data.columns
    })
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, dtype=int)
    missing_cols = set(training_columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0
    data = data[training_columns]
    return data

# UI Setup
st.header("Autism Screening Tool")
st.markdown("""
    This tool predicts Autism Spectrum Disorder (ASD) using a machine learning model.  
    - **Behavioral Scores**: Answer 10 questions (Yes = 1, No = 0) about observed behaviors.
    - **Patient Details**: Provide demographic and medical history.
    - Submit to see the prediction. Results are for research purposes only.
""")
st.markdown("<style>.stForm {border: 1px solid #ddd; padding: 25px; border-radius: 10px; margin-top: 10px;}</style>", unsafe_allow_html=True)

with st.form("patient_form"):
    st.markdown("---")
    st.subheader("Behavioral Scores")
    st.markdown("Answer the following questions about the individual's behavior (Yes = 1, No = 0).")
    
    col1, col2 = st.columns(2)
    with col1:
        a1 = st.selectbox("A1: Notices small details others miss (e.g., patterns, sounds)?", 
                          ["No", "Yes"], index=0, key="a1")
        a2 = st.selectbox("A2: Prefers repetitive routines, upset by changes?", 
                          ["No", "Yes"], index=0, key="a2")
        a3 = st.selectbox("A3: Struggles to understand others’ feelings?", 
                          ["No", "Yes"], index=0, key="a3")
        a4 = st.selectbox("A4: Finds conversations hard to maintain?", 
                          ["No", "Yes"], index=0, key="a4")
        a5 = st.selectbox("A5: Has intense interest in specific topics?", 
                          ["No", "Yes"], index=0, key="a5")
    with col2:
        a6 = st.selectbox("A6: Has difficulty making friends?", 
                          ["No", "Yes"], index=0, key="a6")
        a7 = st.selectbox("A7: Engages in repetitive movements (e.g., hand-flapping)?", 
                          ["No", "Yes"], index=0, key="a7")
        a8 = st.selectbox("A8: Finds social situations confusing?", 
                          ["No", "Yes"], index=0, key="a8")
        a9 = st.selectbox("A9: Has sensitivity to sounds, lights, or textures?", 
                          ["No", "Yes"], index=0, key="a9")
        a10 = st.selectbox("A10: Struggles to interpret facial expressions?", 
                           ["No", "Yes"], index=0, key="a10")
    
    st.markdown("---")
    st.subheader("Patient Details")
    st.markdown("Provide demographic and medical history information below.")
    col3, col4 = st.columns(2)
    with col3:
        # Age input with session state
        if 'age' not in st.session_state:
            st.session_state.age = None
        age_input = st.number_input("Age", min_value=1, max_value=100, value=st.session_state.age, 
                                   format="%d", key="age_input")
        if age_input != st.session_state.age:
            st.session_state.age = age_input
        gender = st.selectbox("Gender", ['Male', 'Female'], format_func=lambda x: x)
        ethnicity = st.selectbox("Ethnicity", [
            "White-European", "Asian", "Black", "Hispanic", "Latino", 
            "Middle Eastern", "Others", "Pasifika", "South Asian", "Turkish"
        ])
        jaundice = st.selectbox("Born with Jaundice", ['No', 'Yes'])
    with col4:
        country = st.selectbox("Country of Residence", [
            "United States", "United Kingdom", "India", "Brazil", "Australia",
            "Canada", "New Zealand", "Other"
        ])
        app_before = st.selectbox("Previously Used App", ['No', 'Yes'])
        relation = st.selectbox("Relation to Patient", [
            "Self", "Parent", "Health care professional", "Relative", "Others"
        ], index=0)
        austim = st.selectbox("Family History of Autism", ['No', 'Yes'])

    submit = st.form_submit_button("Predict")

if submit:
    required_fields = {
        'Age': st.session_state.age, 'Ethnicity': ethnicity, 'Country': country, 'Relation': relation
    }
    empty_fields = [k for k, v in required_fields.items() if v is None or v == '']
    if empty_fields:
        st.error(f"Please fill in: {', '.join(empty_fields)}")
    elif st.session_state.age <= 0:
        st.error("Age must be greater than 0.")
    else:
        # Derive age_desc based on age
        age_desc = "18 and more" if st.session_state.age >= 18 else "Under 18"
        
        patient_data = pd.DataFrame({
            'A1_Score': [1 if a1 == "Yes" else 0],
            'A2_Score': [1 if a2 == "Yes" else 0],
            'A3_Score': [1 if a3 == "Yes" else 0],
            'A4_Score': [1 if a4 == "Yes" else 0],
            'A5_Score': [1 if a5 == "Yes" else 0],
            'A6_Score': [1 if a6 == "Yes" else 0],
            'A7_Score': [1 if a7 == "Yes" else 0],
            'A8_Score': [1 if a8 == "Yes" else 0],
            'A9_Score': [1 if a9 == "Yes" else 0],
            'A10_Score': [1 if a10 == "Yes" else 0],
            'age': [st.session_state.age], 'gender': [gender], 'ethnicity': [ethnicity],
            'jundice': [jaundice], 'austim': [austim], 'country_of_res': [country],
            'used_app_before': [app_before], 'age_desc': [age_desc], 'relation': [relation]
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
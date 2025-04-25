import streamlit as st
import numpy as np
import joblib
from datetime import datetime

# App configuration
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model with caching
@st.cache_resource
def load_model():
    loaded_model = joblib.load('random_forest_model.pkl')
    return loaded_model
    
model = load_model()

# Feature order based on the model
FEATURES = [
    'Age', 'Sex', 'Job', 'Duration', 'Saving accounts_unknown', 
    'Saving accounts_little', 'Saving accounts_quite rich',
    'Saving accounts_rich', 'Saving accounts_moderate',
    'Checking account_little', 'Checking account_moderate',
    'Checking account_unknown', 'Checking account_rich', 'Housing_own',
    'Housing_free', 'Housing_rent', 'Purpose_radio/TV', 'Purpose_education',
    'Purpose_furniture/equipment', 'Purpose_car', 'Purpose_business',
    'Purpose_domestic appliances', 'Purpose_repairs',
    'Purpose_vacation/others'
]

# Custom CSS with improved colors
st.markdown("""
    <style>
    /* Main container styles */
    .main {
        background-color: #f0f2f6;
        padding: 0;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    
    /* Header styling */
    .header-container {
        background-color: #1e3a8a;
        padding: 2rem 1.5rem;
        border-radius: 0 0 10px 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Form container styling */
    .form-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin: 0 0.5rem 1.5rem 0.5rem;
    }
    
    /* Section styling */
    .section {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #2563eb;
    }
    
    /* Section headers */
    .section-header {
        color: #1e3a8a;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6rem 2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.3);
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.4);
    }
    
    /* Result container */
    .result-container {
        text-align: center;
        padding: 2rem;
        border-radius: 8px;
        margin: 1rem 0.5rem;
    }
    
    /* Success result */
    .success-result {
        background-color: #ecfdf5;
        border: 1px solid #d1fae5;
        color: #065f46;
    }
    
    /* Denied result */
    .danger-result {
        background-color: #fef2f2;
        border: 1px solid #fee2e2;
        color: #991b1b;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #4b5563;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    
    /* Help text */
    .help-text {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
    
    /* Card containers */
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Factor items */
    .factor-item {
        background-color: #f0f9ff;
        border-left: 3px solid #2563eb;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-radius: 0 4px 4px 0;
    }
    
    /* Suggestion items */
    .suggestion-item {
        background-color: #fff7ed;
        border-left: 3px solid #ea580c;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-radius: 0 4px 4px 0;
    }
    
    /* Confidence gauge */
    .confidence-container {
        margin: 1.5rem 0;
        text-align: center;
    }
    
    .confidence-gauge {
        display: inline-block;
        width: 180px;
        height: 90px;
        border-radius: 90px 90px 0 0;
        background-color: #e5e7eb;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #2563eb;
        border-radius: 0 0 0 0;
    }
    
    .confidence-label {
        position: absolute;
        bottom: 5px;
        left: 0;
        width: 100%;
        text-align: center;
        color: white;
        font-weight: bold;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Progress steps */
    .steps-container {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0.5rem;
    }
    
    .step {
        flex: 1;
        text-align: center;
        position: relative;
    }
    
    .step-number {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: #e5e7eb;
        color: #6b7280;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        font-weight: bold;
        position: relative;
        z-index: 2;
    }
    
    .step.active .step-number {
        background-color: #2563eb;
        color: white;
    }
    
    .step-label {
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #6b7280;
    }
    
    .step.active .step-label {
        color: #2563eb;
        font-weight: 500;
    }
    
    .step-line {
        position: absolute;
        top: 15px;
        width: 100%;
        height: 2px;
        background-color: #e5e7eb;
        left: -50%;
    }
    
    .step:first-child .step-line {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
    <div class="header-container">
        <h1>💰 Credit Risk Prediction Tool</h1>
        <p>Advanced assessment powered by machine learning</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.title("💼 Credit Risk Assessment")
    st.markdown("""
    This tool helps financial institutions evaluate loan applications using AI-powered risk assessment.
    
    **Features:**
    - Quick risk evaluation
    - Data-driven decision making
    - Consistent assessment criteria
    """)
    
    st.divider()
    with st.expander("About the model"):
        st.markdown("""
        **Model Details:**
        - Algorithm: Random Forest Classifier
        - Training data: German Credit Dataset
        - Key factors considered: age, income, account history, and more
        
        **Note:** This is a demonstration model and should be used with human oversight.
        """)
    
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y')}")

# Steps indicator
st.markdown("""
    <div class="steps-container">
        <div class="step active">
            <div class="step-number">1</div>
            <div class="step-line"></div>
            <div class="step-label">Personal Details</div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-line"></div>
            <div class="step-label">Financial Info</div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-line"></div>
            <div class="step-label">Results</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main form
with st.form("credit_form"):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Personal Information Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
    
    job = st.select_slider("Employment Level", options=[0, 1, 2, 3], 
                         value=2,
                         format_func=lambda x: ["Unskilled & Non-Resident", "Unskilled & Resident", 
                                              "Skilled", "Highly Skilled"][x])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Financial Information Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💵 Financial Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        saving_accounts = st.selectbox("Savings Account Balance", 
                                     ["unknown", "little", "moderate", "quite rich", "rich"],
                                     index=2)
    with col2:
        checking_account = st.selectbox("Checking Account Balance", 
                                      ["unknown", "little", "moderate", "rich"],
                                      index=1)
    
    duration = st.slider("Loan Duration (months)", 
                       min_value=1, max_value=120, value=24)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Information Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏠 Additional Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        housing = st.radio("Housing Status", ["own", "rent", "free"], 
                         horizontal=True)
    with col2:
        purpose = st.selectbox("Loan Purpose", [
            "car", "furniture/equipment", "radio/TV", "domestic appliances",
            "repairs", "education", "business", "vacation/others"
        ])
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close form container
    
    # Submit button
    submit = st.form_submit_button("Assess Credit Risk")

# Processing and results
if submit:
    # Show active step 3
    st.markdown("""
        <div class="steps-container">
            <div class="step">
                <div class="step-number">1</div>
                <div class="step-line"></div>
                <div class="step-label">Personal Details</div>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <div class="step-line"></div>
                <div class="step-label">Financial Info</div>
            </div>
            <div class="step active">
                <div class="step-number">3</div>
                <div class="step-line"></div>
                <div class="step-label">Results</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Processing application data..."):
        # Progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            if i == 33 or i == 66:
                import time
                time.sleep(0.03)
        
        # Initialize feature vector with zeros
        input_vector = np.zeros(len(FEATURES))

        # Assign numerical values
        input_vector[FEATURES.index('Age')] = age
        input_vector[FEATURES.index('Sex')] = 1 if sex == "Male" else 0
        input_vector[FEATURES.index('Job')] = job
        input_vector[FEATURES.index('Duration')] = duration

        # One-hot encoding for Saving accounts
        save_col = f"Saving accounts_{saving_accounts}"
        if save_col in FEATURES:
            input_vector[FEATURES.index(save_col)] = 1

        # One-hot encoding for Checking account
        check_col = f"Checking account_{checking_account}"
        if check_col in FEATURES:
            input_vector[FEATURES.index(check_col)] = 1

        # One-hot encoding for Housing
        house_col = f"Housing_{housing}"
        if house_col in FEATURES:
            input_vector[FEATURES.index(house_col)] = 1

        # One-hot encoding for Purpose
        purpose_col = f"Purpose_{purpose}"
        if purpose_col in FEATURES:
            input_vector[FEATURES.index(purpose_col)] = 1

        # Make prediction
        prediction = model.predict([input_vector])[0]
        
        # Calculate confidence (for UI demonstration)
        confidence = model.predict_proba([input_vector])[0]
        confidence_value = confidence[1] if prediction == 1 else confidence[0]
        confidence_percentage = round(confidence_value * 100)
        
    # Clear the progress bar
    progress_bar.empty()
    
    # Display results
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f"""
            <div class="result-container success-result">
                <h2>✅ Credit Application Approved</h2>
                <p>Based on the provided information, this application has been assessed as a good credit risk.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence gauge
        st.markdown(f"""
            <div class="confidence-container">
                <p>Confidence Assessment</p>
                <div class="confidence-gauge">
                    <div class="confidence-fill" style="height: {confidence_percentage}%;"></div>
                    <div class="confidence-label">{confidence_percentage}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Contributing factors
        st.subheader("Key Positive Factors")
        
        factors = []
        if age > 30:
            factors.append("Age indicates established financial history")
        if job >= 2:
            factors.append("Skilled employment status")
        if saving_accounts in ["moderate", "quite rich", "rich"]:
            factors.append("Good savings account balance")
        if checking_account in ["moderate", "rich"]:
            factors.append("Healthy checking account")
        if housing == "own":
            factors.append("Property ownership")
        if duration <= 24:
            factors.append("Short-term loan duration")
            
        for factor in factors[:4]:  # Limit to top 4 factors
            st.markdown(f"""<div class="factor-item">{factor}</div>""", unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
            <div class="result-container danger-result">
                <h2>❌ Credit Application Not Approved</h2>
                <p>Based on the provided information, this application has been assessed as a higher credit risk.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence gauge
        st.markdown(f"""
            <div class="confidence-container">
                <p>Risk Assessment Confidence</p>
                <div class="confidence-gauge">
                    <div class="confidence-fill" style="height: {confidence_percentage}%;"></div>
                    <div class="confidence-label">{confidence_percentage}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Suggestions for improvement
        st.subheader("Suggestions for Improvement")
        
        suggestions = []
        if duration > 36:
            suggestions.append("Consider a shorter loan duration")
        if saving_accounts in ["unknown", "little"]:
            suggestions.append("Improve savings account balance")
        if checking_account in ["unknown", "little"]:
            suggestions.append("Build a stronger checking account history")
        if job < 2:
            suggestions.append("Higher employment qualification may help")
        if age < 25:
            suggestions.append("Longer credit history would strengthen application")
        
        for suggestion in suggestions[:4]:  # Limit to top 4 suggestions
            st.markdown(f"""<div class="suggestion-item">{suggestion}</div>""", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close card container

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2025 Credit Risk Assessment Tool</p>
        <p>This is a demonstration application for educational purposes only.</p>
    </div>
""", unsafe_allow_html=True)

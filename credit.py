import streamlit as st
import numpy as np
import joblib
from datetime import datetime

# App configuration
st.set_page_config(
    page_title="Credit Risk Predictor",
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

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main container styles */
    .main {
        background-color: #f8f9fa;
        padding: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #3494e6, #ec6ead);
        padding: 2rem 1.5rem;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Form container styling */
    .form-container {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        margin: 0 1rem 2rem 1rem;
    }
    
    /* Section styling */
    .section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3494e6;
    }
    
    /* Section headers */
    .section-header {
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #3494e6, #4a6bdf);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        border: none;
        box-shadow: 0 4px 10px rgba(52, 148, 230, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        box-shadow: 0 6px 15px rgba(52, 148, 230, 0.5);
        transform: translateY(-2px);
    }
    
    /* Result container */
    .result-container {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Success result */
    .success-result {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    /* Denied result */
    .danger-result {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    /* Slider styling */
    .stSlider div[data-baseweb="slider"] div {
        background-color: #3494e6 !important;
    }
    
    /* Help text */
    .help-text {
        font-size: 0.85rem;
        color: #6c757d;
        font-style: italic;
        margin-top: 0.3rem;
    }
    
    /* Label emphasis */
    label {
        font-weight: 500 !important;
        color: #2c3e50 !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Progress indicator */
    .progress-indicator {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    
    .progress-step {
        flex: 1;
        text-align: center;
        padding: 0.5rem;
        border-bottom: 3px solid #dee2e6;
        color: #6c757d;
        font-weight: 500;
    }
    
    .progress-step.active {
        border-bottom: 3px solid #3494e6;
        color: #3494e6;
    }
    </style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
    <div class="header-container">
        <h1>💰 Smart Credit Risk Predictor</h1>
        <p>AI-powered credit assessment for informed lending decisions</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.image("https://www.svgrepo.com/show/81080/analytics.svg", width=100)
    st.title("About this app")
    st.markdown("""
    This application uses machine learning to predict the risk associated with providing a loan to an applicant.
    
    **How it works:**
    1. Enter applicant information
    2. Submit for AI analysis
    3. Review the risk assessment
    
    **Model information:**
    - Algorithm: Random Forest
    - Accuracy: ~76%
    - Last updated: April 2025
    """)
    
    st.divider()
    st.markdown(f"**Current date:** {datetime.now().strftime('%B %d, %Y')}")
    st.markdown("**Version:** 2.1.0")

# Main form
with st.form("credit_form"):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Personal Information Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30, help="Applicant's age in years")
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"], 
                         format_func=lambda x: {"Male": "Male", "Female": "Female"}[x])
    
    job = st.select_slider("Employment Level", options=[0, 1, 2, 3], 
                         value=2,
                         format_func=lambda x: ["Unskilled & Non-Resident", "Unskilled & Resident", 
                                              "Skilled", "Highly Skilled"][x])
    st.markdown('<p class="help-text">Higher employment levels may indicate more stable income sources</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Financial Information Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💵 Financial Information</div>', unsafe_allow_html=True)
    
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
                       min_value=1, max_value=120, value=24, 
                       help="The period for loan repayment in months")
    
    st.markdown('<p class="help-text">Shorter loan durations typically have lower risk profiles</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Information Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏠 Additional Information</div>', unsafe_allow_html=True)
    
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
    submit = st.form_submit_button("Analyze Credit Risk")

# Processing and results
if submit:
    with st.spinner("Analyzing applicant data..."):
        # Create progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            # Update progress bar
            progress_bar.progress(i + 1)
            if i == 25 or i == 50 or i == 75:
                # Simulate processing time
                import time
                time.sleep(0.05)
        
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
        
        # Calculate confidence (this is just for UI demonstration)
        # In a real app, you'd use model.predict_proba() for actual probabilities
        confidence = model.predict_proba([input_vector])[0]
        confidence_value = confidence[1] if prediction == 1 else confidence[0]
        confidence_percentage = round(confidence_value * 100, 1)
        
    # Clear the progress bar after completion
    progress_bar.empty()
    
    # Display results with different styles based on the prediction
    if prediction == 1:
        st.markdown(f"""
            <div class="result-container success-result">
                <h2>✅ Credit Application Approved</h2>
                <p>The applicant has been assessed as a good credit risk.</p>
                <h3>Confidence: {confidence_percentage}%</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Show factors that contributed to approval
        st.subheader("Positive Contributing Factors")
        factors = []
        if age > 30:
            factors.append("Age indicates financial stability")
        if job >= 2:
            factors.append("Skilled employment status")
        if saving_accounts in ["moderate", "quite rich", "rich"]:
            factors.append("Good savings account balance")
        if checking_account in ["moderate", "rich"]:
            factors.append("Healthy checking account")
        if housing == "own":
            factors.append("Property ownership")
        if duration <= 24:
            factors.append("Reasonable loan duration")
            
        # Display factors or default message
        if factors:
            for i, factor in enumerate(factors, 1):
                st.markdown(f"**{i}.** {factor}")
        else:
            st.markdown("Multiple factors contributed to this decision")
    else:
        st.markdown(f"""
            <div class="result-container danger-result">
                <h2>❌ Credit Application Not Approved</h2>
                <p>The applicant has been assessed as a higher credit risk.</p>
                <h3>Confidence: {confidence_percentage}%</h3>
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
        
        # Display suggestions or default message
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"**{i}.** {suggestion}")
        else:
            st.markdown("Consider reapplying with improved financial indicators")

# Footer
st.markdown("""
    <div class="footer">
        <p>&copy; 2025 Credit Risk Predictor | Powered by Machine Learning</p>
        <p>Disclaimer: This is a demonstration application and should not be used for actual financial decisions.</p>
    </div>
""", unsafe_allow_html=True)
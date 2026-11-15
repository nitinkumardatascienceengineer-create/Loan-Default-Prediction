
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("best_random_forest_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Default Prediction")
st.write(
    "Enter the borrower's financial and personal details "
    "to predict the possibility of loan default."
)

st.divider()

# Input form
with st.form("loan_prediction_form"):

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)

        employment_type = st.text_input(
            "Employment Type",
            value="Salaried",
            help="Use a category exactly as it appears in your training dataset."
        )

    with col2:
        employment_years = st.number_input(
            "Employment Years", min_value=0.0, value=5.0
        )

        education = st.text_input(
            "Education",
            value="Graduate",
            help="Use a category exactly as it appears in your training dataset."
        )

    with col3:
        dependents = st.number_input(
            "Dependents", min_value=0.0, value=1.0
        )

    st.divider()
    st.subheader("Income and Financial Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        annual_income = st.number_input(
            "Annual Income (₹)", min_value=0.0, value=600000.0
        )

        monthly_income = st.number_input(
            "Monthly Income (₹)", min_value=0, value=50000
        )

        savings_balance = st.number_input(
            "Savings Balance (₹)", min_value=0.0, value=100000.0
        )

    with col2:
        existing_loan_amount = st.number_input(
            "Existing Loan Amount (₹)", min_value=0, value=100000
        )

        existing_emi = st.number_input(
            "Existing EMI (₹)", min_value=0, value=5000
        )

        credit_score = st.number_input(
            "Credit Score", min_value=0.0, value=700.0
        )

    with col3:
        debt_to_income_ratio = st.number_input(
            "Debt-to-Income Ratio", min_value=0.0, max_value=1.0,
            value=0.30
        )

        loan_amount = st.number_input(
            "Requested Loan Amount (₹)", min_value=0, value=500000
        )

    st.divider()
    st.subheader("Loan Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_term_months = st.number_input(
            "Loan Term (Months)", min_value=1, value=60
        )

        interest_rate = st.number_input(
            "Interest Rate (%)", min_value=0.0, value=10.0
        )

    with col2:
        loan_purpose = st.text_input(
            "Loan Purpose",
            value="Home",
            help="Use a category exactly as it appears in your training dataset."
        )

        collateral_value = st.number_input(
            "Collateral Value (₹)", min_value=0.0, value=500000.0
        )

    with col3:
        previous_loan_count = st.number_input(
            "Previous Loan Count", min_value=0, value=1
        )

        previous_defaults = st.number_input(
            "Previous Defaults", min_value=0, value=0
        )

    st.divider()
    st.subheader("Credit History")

    col1, col2, col3 = st.columns(3)

    with col1:
        late_payments_12m = st.number_input(
            "Late Payments (Last 12 Months)",
            min_value=0, value=0
        )

    with col2:
        credit_card_utilization = st.number_input(
            "Credit Card Utilization",
            min_value=0.0, max_value=1.0, value=0.30
        )

    with col3:
        credit_history_years = st.number_input(
            "Credit History (Years)", min_value=0, value=5
        )

    st.divider()

    submit_button = st.form_submit_button(
        "Predict Loan Default",
        use_container_width=True
    )


# Prediction
if submit_button:

    input_data = pd.DataFrame([{
        "Age": age,
        "Employment_Type": employment_type,
        "Employment_Years": employment_years,
        "Education": education,
        "Dependents": dependents,
        "Annual_Income": annual_income,
        "Monthly_Income": monthly_income,
        "Existing_Loan_Amount": existing_loan_amount,
        "Existing_EMI": existing_emi,
        "Credit_Score": credit_score,
        "Savings_Balance": savings_balance,
        "Debt_to_Income_Ratio": debt_to_income_ratio,
        "Loan_Amount": loan_amount,
        "Loan_Term_Months": loan_term_months,
        "Interest_Rate": interest_rate,
        "Loan_Purpose": loan_purpose,
        "Collateral_Value": collateral_value,
        "Previous_Loan_Count": previous_loan_count,
        "Previous_Defaults": previous_defaults,
        "Late_Payments_12M": late_payments_12m,
        "Credit_Card_Utilization": credit_card_utilization,
        "Credit_History_Years": credit_history_years
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ High Default Risk")
            st.write("The model predicts that this borrower may default.")
        else:
            st.success("✅ Low Default Risk")
            st.write("The model predicts that this borrower may not default.")

    with col2:
        st.metric(
            "Predicted Default Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

    st.caption(
        "This is a machine learning prediction, not a final lending decision."
    )

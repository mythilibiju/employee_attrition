import streamlit as st
import pandas as pd
import pickle

# Load artifacts
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoder.pkl", "rb"))

st.title("Employee Attrition Prediction System")

st.sidebar.header("Enter Employee Details")

def user_input():
    data = {}

    # ---- NUMERICAL FEATURES ----
    data["Age"] = st.sidebar.slider("Age", 18, 60, 30)
    data["DailyRate"] = st.sidebar.number_input("Daily Rate", 100, 1500, 500)
    data["DistanceFromHome"] = st.sidebar.slider("Distance From Home", 1, 30, 5)
    data["HourlyRate"] = st.sidebar.slider("Hourly Rate", 30, 100, 60)
    data["MonthlyIncome"] = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
    data["MonthlyRate"] = st.sidebar.number_input("Monthly Rate", 1000, 30000, 15000)
    data["NumCompaniesWorked"] = st.sidebar.slider("Companies Worked", 0, 10, 2)
    data["PercentSalaryHike"] = st.sidebar.slider("Salary Hike (%)", 10, 25, 15)
    data["TotalWorkingYears"] = st.sidebar.slider("Total Working Years", 0, 40, 5)
    data["TrainingTimesLastYear"] = st.sidebar.slider("Training Times Last Year", 0, 10, 2)
    data["YearsAtCompany"] = st.sidebar.slider("Years At Company", 0, 40, 5)
    data["YearsInCurrentRole"] = st.sidebar.slider("Years In Current Role", 0, 20, 3)
    data["YearsSinceLastPromotion"] = st.sidebar.slider("Years Since Last Promotion", 0, 15, 1)
    data["YearsWithCurrManager"] = st.sidebar.slider("Years With Current Manager", 0, 20, 3)
    data["StockOptionLevel"] = st.sidebar.selectbox("Stock Option Level", [0,1,2,3])

    # ---- ORDINAL FEATURES ----
    data["Education"] = st.sidebar.selectbox("Education", [1,2,3,4,5])
    data["EnvironmentSatisfaction"] = st.sidebar.selectbox("Environment Satisfaction", [1,2,3,4])
    data["JobInvolvement"] = st.sidebar.selectbox("Job Involvement", [1,2,3,4])
    data["JobSatisfaction"] = st.sidebar.selectbox("Job Satisfaction", [1,2,3,4])
    data["PerformanceRating"] = st.sidebar.selectbox("Performance Rating", [1,2,3,4])
    data["RelationshipSatisfaction"] = st.sidebar.selectbox("Relationship Satisfaction", [1,2,3,4])
    data["WorkLifeBalance"] = st.sidebar.selectbox("Work Life Balance", [1,2,3,4])
    data["JobLevel"] = st.sidebar.selectbox("Job Level", [1,2,3,4,5])

    # ---- CATEGORICAL FEATURES ----
    data["BusinessTravel"] = st.sidebar.selectbox("Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    data["Department"] = st.sidebar.selectbox("Department",
        ["Sales", "Research & Development", "Human Resources"])

    data["EducationField"] = st.sidebar.selectbox("Education Field",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])

    data["Gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])

    data["JobRole"] = st.sidebar.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative",
        "Manager", "Sales Representative", "Research Director", "Human Resources"
    ])

    data["MaritalStatus"] = st.sidebar.selectbox("Marital Status",
        ["Single", "Married", "Divorced"])

    data["OverTime"] = st.sidebar.selectbox("OverTime", ["Yes", "No"])

    return pd.DataFrame([data])

input_df = user_input()

# ---- ENCODE ----
for col in input_df.columns:
    if col in encoders:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])

# ---- ENSURE COLUMN ORDER MATCHES TRAINING ----
feature_order = scaler.feature_names_in_
input_df = input_df[feature_order]

# ---- SCALE ----
input_scaled = scaler.transform(input_df)

# ---- PREDICT ----
prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

# ---- OUTPUT ----
st.subheader("Prediction Result")

if prediction == 1:
    st.error(f"⚠️ High Risk of Attrition ({prob:.2f})")
else:
    st.success(f"✅ Low Risk of Attrition ({prob:.2f})")
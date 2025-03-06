import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import streamlit as st
import plotly.express as px

url="https://raw.githubusercontent.com/Divyaprasad-Sahu/HR_ATTRITION/refs/heads/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(url)

df["Attrition"] = df["Attrition"].map({"No": 0, "Yes": 1})
df["BusinessTravel"] = df["BusinessTravel"].map({"Non-Travel": 1, "Travel_Rarely": 2, "Travel_Frequently": 3})
df["Department"] = df["Department"].map({"Human Resources": 1, "IT": 2, "Finance": 3, "Sales": 4, "Research & Development": 5})
df["EducationField"] = df["EducationField"].map({"Human Resources": 1, "Life Sciences": 2, "Marketing": 3,"Medical": 4, "Other": 5, "Technical Degree": 6})
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 2})
df["JobRole"] = df["JobRole"].map({
    "Healthcare Representative": 1, "Human Resources": 2, "Laboratory Technician": 3, "Manager": 4,
    "Manufacturing Director": 5, "Research Director": 6, "Research Scientist": 7, "Sales Executive": 8, "Sales Representative": 9
})
df["MaritalStatus"] = df["MaritalStatus"].map({"Single": 1, "Married": 2, "Divorced": 3})
df["OverTime"] = df["OverTime"].map({"No": 0, "Yes": 1})

df.drop(columns=["Over18", "EmployeeCount", "EmployeeNumber", "StandardHours"], inplace=True)


X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

joblib.dump(model, "attrition_model.pkl")
joblib.dump(scaler, "scaler.pkl")
model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
st.title("Employee Attrition Prediction & Retention Strategies")
st.markdown("Enter employee details to predict attrition and get HR retention strategies.")

age = st.number_input("Age", min_value=18, max_value=65, value=30)
business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel Rarely", "Travel Frequently"])
daily_rate = st.number_input("Daily Rate", min_value=100, max_value=2000, value=500)
department = st.selectbox("Department", ["Human Resources", "IT", "Finance", "Sales", "Research & Development"])
distance_from_home = st.number_input("Distance From Home", min_value=0, max_value=50, value=5)
education = st.slider("Education Level", 1, 5, 3)
education_field = st.selectbox("Education Field", ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])
env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
gender = st.selectbox("Gender", ["Male", "Female"])
hourly_rate = st.number_input("Hourly Rate", min_value=10, max_value=100, value=30)
job_involvement = st.slider("Job Involvement", 1, 4, 3)
job_level = st.slider("Job Level", 1, 5, 3)
job_role = st.selectbox("Job Role", [
    "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
    "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"])
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
monthly_rate = st.number_input("Monthly Rate", min_value=1000, max_value=50000, value=10000)
num_companies_worked = st.number_input("Num Companies Worked", min_value=0, max_value=10, value=3)
overtime = st.selectbox("Overtime", ["No", "Yes"])
percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=50, value=10)
performance_rating = st.slider("Performance Rating", 1, 5, 3)
relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=10)
training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=2)
work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
years_at_company = st.number_input("Years At Company", min_value=0, max_value=50, value=5)
years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=3)
years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=2)
years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=4)


def encode(value, mapping):
    return mapping.get(value, 0)
encoded_values = np.array([[  
    age,
    encode(business_travel, {"Non-Travel": 1, "Travel Rarely": 2, "Travel Frequently": 3}),
    daily_rate,
    encode(department, {"Human Resources": 1, "IT": 2, "Finance": 3, "Sales": 4, "Research & Development": 5}),
    distance_from_home,
    education,
    encode(education_field, {"Human Resources": 1, "Life Sciences": 2, "Marketing": 3, "Medical": 4, "Other": 5, "Technical Degree": 6}),
    env_satisfaction,
    encode(gender, {"Male": 1, "Female": 2}),
    hourly_rate,
    job_involvement,
    job_level,
    encode(job_role, {
        "Healthcare Representative": 1, "Human Resources": 2, "Laboratory Technician": 3, "Manager": 4,
        "Manufacturing Director": 5, "Research Director": 6, "Research Scientist": 7, "Sales Executive": 8, "Sales Representative": 9}),
    job_satisfaction,
    encode(marital_status, {"Single": 1, "Married": 2, "Divorced": 3}),
    monthly_income,
    monthly_rate,
    num_companies_worked,
    encode(overtime, {"No": 0, "Yes": 1}),
    percent_salary_hike,
    performance_rating,
    relationship_satisfaction,
    stock_option_level,
    total_working_years,
    training_times_last_year,
    work_life_balance,
    years_at_company,
    years_in_current_role,
    years_since_last_promotion,
    years_with_curr_manager
]])
encoded_values_scaled = scaler.transform(encoded_values)
if st.button("Predict"):
    prediction = model.predict(encoded_values_scaled)
    probability = model.predict_proba(encoded_values_scaled)

    if prediction[0] == 1:
        st.subheader("Likely to Leave")
        st.write(f"Probability of Leaving: {probability[0][1]:.2f}")
        
        st.markdown("### HR Retention Strategies:")
        
        if job_satisfaction < 3:
            st.write("**Boost Job Satisfaction** → Provide recognition, clear career growth, and training programs.")
        if work_life_balance < 3:
            st.write(" **Enhance Work-Life Balance** → Offer flexible hours, hybrid work options, and reduced overtime.")
        if monthly_income < 5000:
            st.write("**Increase Compensation** → Consider salary adjustments and performance-based bonuses.")
        if years_since_last_promotion > 3:
            st.write("**Plan Promotions** → Provide internal growth opportunities and skill development plans.")
        if env_satisfaction < 3:
            st.write("**Improve Workplace Culture** → Address concerns through HR surveys and mental health initiatives.")
        if overtime == "Yes":
            st.write("**Reduce Overtime** → Optimize workload distribution and encourage work-life balance.")
        if training_times_last_year < 2:
            st.write("**Increase Training** → Provide leadership training, new skills, and mentorship programs.")
        if relationship_satisfaction < 3:
            st.write("**Improve Manager-Employee Relations** → Encourage open feedback and coaching sessions.")
    else:
        st.subheader("Likely to Stay")
        st.write(f"Probability of Staying: {probability[0][0]:.2f}")
avg_age = df["Age"].mean()
avg_monthly_income = df["MonthlyIncome"].mean()
avg_job_satisfaction = df["JobSatisfaction"].mean()
avg_work_life_balance = df["WorkLifeBalance"].mean()
avg_years_since_last_promotion = df["YearsSinceLastPromotion"].mean()
avg_total_working_years = df["TotalWorkingYears"].mean()
if st.button("Visualize"):
    st.markdown("### 🔍 Employee vs. Company Average")
    
    metrics = ["Age", "Monthly Income", "Job Satisfaction", "Work-Life Balance", "Years Since Last Promotion", "Total Working Years"]
    employee_values = [age, monthly_income, job_satisfaction, work_life_balance, years_since_last_promotion, total_working_years]
    avg_values = [avg_age, avg_monthly_income, avg_job_satisfaction, avg_work_life_balance, avg_years_since_last_promotion, avg_total_working_years]
    
    for metric, emp_val, avg_val in zip(metrics, employee_values, avg_values):
        comparison_df = pd.DataFrame({"Category": ["Employee", "Company Average"], "Value": [emp_val, avg_val]})
        fig = px.bar(comparison_df, x="Category", y="Value", title=f"{metric} Comparison", color="Category")
        st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import joblib

# Load model & encoders
model = joblib.load("dev_recommender_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Hybrid prediction function (hard rule included)
def predict_with_rules(sample):
    if sample["dev_on_leave"] == True:
        return 0
    temp = pd.DataFrame([sample])
    for col in temp.columns:
        if col in label_encoders:
            temp[col] = label_encoders[col].transform(temp[col])
    return model.predict(temp)[0]

# Streamlit UI
st.title("Developer Task Recommender")

project_type = st.selectbox("Project Type", ["app", "web", "game"])
required_seniority = st.selectbox("Required Seniority", ["junior", "mid", "senior"])
dev_specialty = st.selectbox("Developer Specialty", ["app", "web", "game"])
dev_seniority = st.selectbox("Developer Seniority", ["junior", "mid", "senior"])
dev_workload = st.selectbox("Developer Workload", ["free", "light", "heavy"])
dev_on_leave = st.checkbox("Developer on Leave?")
dev_tasks_this_week = st.number_input("Tasks Assigned This Week", 0, 10, 0)

input_data = {
    "project_type": project_type,
    "required_seniority": required_seniority,
    "dev_specialty": dev_specialty,
    "dev_seniority": dev_seniority,
    "dev_workload": dev_workload,
    "dev_on_leave": dev_on_leave,
    "dev_tasks_this_week": dev_tasks_this_week
}

if st.button("Check Recommendation"):
    result = predict_with_rules(input_data)
    st.success("✅ Recommended" if result == 1 else "❌ Not Recommended")

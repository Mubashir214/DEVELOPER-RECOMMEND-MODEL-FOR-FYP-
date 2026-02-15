# app.py
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model & Label Encoders
# -----------------------------
model = joblib.load("dev_recommender_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# -----------------------------
# Prediction Function with Rules
# -----------------------------
def predict_with_rules(sample: dict) -> int:
    """
    Hybrid prediction:
    - Hard rule: if developer is on leave -> Not recommended
    - Otherwise: ML-based recommendation
    """
    if sample.get("dev_on_leave", False):
        return 0

    temp = pd.DataFrame([sample])
    for col in temp.columns:
        if col in label_encoders:
            temp[col] = label_encoders[col].transform(temp[col])
    
    return int(model.predict(temp)[0])

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Developer Task Recommender", page_icon="💻", layout="centered")

st.title("💻 Developer Task Recommender")
st.markdown("""
Predict whether a developer is suitable for a task based on workload, skills, and availability.
**Note:** Developers on leave will never be recommended.
""")

# Input Form
with st.form("dev_input_form"):
    st.subheader("Developer & Task Information")

    project_type = st.selectbox("Project Type", ["app", "web", "game"])
    required_seniority = st.selectbox("Required Seniority", ["junior", "mid", "senior"])
    dev_specialty = st.selectbox("Developer Specialty", ["app", "web", "game"])
    dev_seniority = st.selectbox("Developer Seniority", ["junior", "mid", "senior"])
    dev_workload = st.selectbox("Developer Workload", ["free", "light", "heavy"])
    dev_on_leave = st.checkbox("Developer on Leave?")
    dev_tasks_this_week = st.number_input("Tasks Assigned This Week", 0, 10, 0)

    submitted = st.form_submit_button("Check Recommendation")

if submitted:
    input_data = {
        "project_type": project_type,
        "required_seniority": required_seniority,
        "dev_specialty": dev_specialty,
        "dev_seniority": dev_seniority,
        "dev_workload": dev_workload,
        "dev_on_leave": dev_on_leave,
        "dev_tasks_this_week": dev_tasks_this_week
    }

    result = predict_with_rules(input_data)

    if result == 1:
        st.success("✅ Developer is Recommended for this Task")
        st.balloons()
    else:
        st.error("❌ Developer is NOT Recommended for this Task")
        st.info("Tip: Ensure the developer is available and workload matches the project requirements.")

# Optional: Show Example Test Cases
if st.checkbox("Show Example Test Cases"):
    test_cases = [
        {
            "project_type": "web", "required_seniority": "mid",
            "dev_specialty": "web", "dev_seniority": "senior",
            "dev_workload": "light", "dev_on_leave": False, "dev_tasks_this_week": 1
        },
        {
            "project_type": "game", "required_seniority": "mid",
            "dev_specialty": "web", "dev_seniority": "senior",
            "dev_workload": "free", "dev_on_leave": False, "dev_tasks_this_week": 0
        },
        {
            "project_type": "app", "required_seniority": "junior",
            "dev_specialty": "app", "dev_seniority": "junior",
            "dev_workload": "free", "dev_on_leave": True, "dev_tasks_this_week": 0
        }
    ]
    st.write(pd.DataFrame(test_cases))

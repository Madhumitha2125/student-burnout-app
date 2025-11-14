import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------------
# Load New Numeric Stress Model
# ------------------------------
model = joblib.load("stress_model.joblib")
scaler = joblib.load("stress_scaler.joblib")

st.set_page_config(
    page_title="Student Stress Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict Single Student", "Report"])

# ------------------------------
# Page 1: Dashboard (Full Dataset)
# ------------------------------
if page == "Dashboard":
    st.title("📊 Student Stress Analysis Dashboard")

    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset Loaded Successfully!")
        st.write("### Preview of Dataset")
        st.dataframe(df)

        st.divider()
        st.write("### 🔍 Predict Stress Level (1–100) for All Students")

        # ---------------------------
        # Convert Categorical Inputs
        # ---------------------------
        df_encoded = df.copy()

        # Map Study Hours
        mapping_study = {
            "Less than 2 hours": 1,
            "2-4 hours": 2,
            "4-6 hours": 3,
            "More than 6 hours": 4
        }

        # Map Sleep Hours
        mapping_sleep = {
            "Less than 4 hours": 1,
            "4-6 hours": 2,
            "6-8 hours": 3,
            "More than 8 hours": 4
        }

        # Map Stress Level
        mapping_stress = {"Low": 1, "Moderate": 2, "High": 3}

        df_encoded["Study Hours"] = df["Study Hours"].map(mapping_study)
        df_encoded["Sleep Hours"] = df["Sleep Hours"].map(mapping_sleep)
        df_encoded["Stress"] = df["Stress"].map(mapping_stress)

        # ---------------------------
        # Prepare features
        # ---------------------------
        features = df_encoded[["CGPA (Out of 10)", "Attendance Percentage", "Study Hours", "Sleep Hours", "Stress"]]

        scaled = scaler.transform(features)
        predictions = model.predict(scaled)

        df["Predicted Stress (1–100)"] = predictions

        st.success("Predictions Completed!")
        st.dataframe(df)

        # ---------------------------
        # Charts
        # ---------------------------
        st.subheader("📈 Stress Level Distribution")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist(predictions, bins=10)
        ax.set_xlabel("Stress Level")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)

        # Download updated CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predicted Dataset",
            csv,
            "predicted_stress.csv",
            "text/csv"
        )

    else:
        st.info("Please upload a CSV file to begin.")

# ------------------------------
# Page 2: Predict Single Student
# ------------------------------
elif page == "Predict Single Student":
    st.title("🧠 Predict Stress for One Student (1–100)")

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.number_input("CGPA (Out of 10)", min_value=0.0, max_value=10.0, step=0.1)
        attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=1.0)

    with col2:
        study_hours = st.selectbox("Study Hours", ["Less than 2 hours", "2-4 hours", "4-6 hours", "More than 6 hours"])
        sleep_hours = st.selectbox("Sleep Hours", ["Less than 4 hours", "4-6 hours", "6-8 hours", "More than 8 hours"])
        stress_prev = st.selectbox("Previous Stress Level (Self-Reported)", ["Low", "Moderate", "High"])

    mapping_study = {"Less than 2 hours": 1, "2-4 hours": 2, "4-6 hours": 3, "More than 6 hours": 4}
    mapping_sleep = {"Less than 4 hours": 1, "4-6 hours": 2, "6-8 hours": 3, "More than 8 hours": 4}
    mapping_stress = {"Low": 1, "Moderate": 2, "High": 3}

    input_data = np.array([[cgpa, attendance,
                            mapping_study[study_hours],
                            mapping_sleep[sleep_hours],
                            mapping_stress[stress_prev]]])

    scaled_input = scaler.transform(input_data)

    if st.button("🎯 Predict Stress"):
        result = model.predict(scaled_input)[0]
        st.success(f"🔥 Predicted Stress Level: **{int(result)} / 100**")

# ------------------------------
# Page 3: Report Page
# ------------------------------
elif page == "Report":
    st.title("📄 Project Report – Student Stress Prediction")

    st.write("""
    ### Project Overview  
    This dashboard predicts student stress levels on a **1–100 numeric scale** using a trained machine learning model.

    ### Inputs Used
    - CGPA  
    - Attendance  
    - Study Hours  
    - Sleep Hours  
    - Self-reported stress (Low/Moderate/High)

    ### Output  
    - A continuous stress score between **1 and 100**, indicating the student's estimated stress intensity.

    ### Applications  
    - Early identification of high-stress students  
    - Academic counselling support  
    - Wellness monitoring  
    """)






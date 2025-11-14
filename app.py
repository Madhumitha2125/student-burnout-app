import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Stress Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# Load the classification model (Low/Moderate/High)
# ------------------------------
model = joblib.load("stress_model.joblib")
scaler = joblib.load("stress_scaler.joblib")

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict Single Student", "Report"])

# ------------------------------
# Page 1: Dashboard (Full Dataset)
# ------------------------------
if page == "Dashboard":
    st.title("📊 Student Stress Category Prediction")

    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset Loaded Successfully!")
        st.write("### Preview of Dataset")
        st.dataframe(df)

        st.divider()
        st.write("### 🔍 Predict Stress Category (Low / Moderate / High)")

        # ---------------------------
        # Label Encoding (matches your dataset)
        # ---------------------------
        mapping_study = {
            "0-2 hrs": 1,
            "3-5 hrs": 2,
            "5-7 hrs": 3,
            "7-9 hrs": 4
        }

        mapping_sleep = {
            "5-6 hrs": 1,
            "7-8 hrs": 2,
            "9-10 hrs": 3
        }

        df_encoded = df.copy()

        df_encoded["Study Hours"] = df["Study Hours"].map(mapping_study)
        df_encoded["Sleep Hours"] = df["Sleep Hours"].map(mapping_sleep)

        # Prepare features
        features = df_encoded[["CGPA (Out of 10)", "Attendance Percentage",
                               "Study Hours", "Sleep Hours"]]

        scaled = scaler.transform(features)
        predictions = model.predict(scaled)

        df["Predicted Stress Category"] = predictions

        st.success("Prediction Completed!")
        st.dataframe(df)

        # ---------------------------
        # Chart
        # ---------------------------
        st.subheader("📈 Stress Category Count")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist(predictions, bins=3)
        ax.set_xlabel("Stress Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Download updated CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predicted Dataset",
            csv,
            "predicted_stress_categories.csv",
            "text/csv"
        )

    else:
        st.info("Please upload a CSV file to begin.")

# ------------------------------
# Page 2: Predict Single Student
# ------------------------------
elif page == "Predict Single Student":
    st.title("🎯 Predict Stress Category for One Student")

    cgpa = st.number_input("CGPA (out of 10)", 0.0, 10.0, 7.0)
    attendance = st.number_input("Attendance Percentage", 0, 100, 75)

    study = st.selectbox("Study Hours", ["0-2 hrs", "3-5 hrs", "5-7 hrs", "7-9 hrs"])
    sleep = st.selectbox("Sleep Hours", ["5-6 hrs", "7-8 hrs", "9-10 hrs"])

    mapping_study = {"0-2 hrs": 1, "3-5 hrs": 2, "5-7 hrs": 3, "7-9 hrs": 4}
    mapping_sleep = {"5-6 hrs": 1, "7-8 hrs": 2, "9-10 hrs": 3}

    if st.button("Predict"):
        X = [[cgpa, attendance, mapping_study[study], mapping_sleep[sleep]]]
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

        st.success(f"Predicted Stress Category: **{pred}**")

# ------------------------------
# Page 3: Report Page
# ------------------------------
elif page == "Report":
    st.title("📄 Project Report – Stress Category Prediction")

    st.write("""
    ### Project Overview  
    This dashboard predicts student stress levels as **Low, Moderate, or High** using a trained classification model.

    ### Inputs Used
    - CGPA  
    - Attendance  
    - Study Hours  
    - Sleep Hours  

    ### Output  
    - Stress Category: **Low / Moderate / High**

    ### Applications  
    - Identifying high-stress students  
    - Academic support planning  
    - Counseling and well-being improvement  
    """)











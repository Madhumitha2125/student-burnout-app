import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Load Model and Scaler ---
model = joblib.load("burnout_model.joblib")
scaler = joblib.load("scaler.joblib")

# --- Navigation Sidebar ---
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Burnout Predictor", "📈 Data Report"])

# ------------------ PAGE 1: BURNOUT PREDICTOR ------------------
if page == "🏠 Burnout Predictor":
    st.title("🎓 Student Burnout Level Prediction App")

    # --- Load Existing Dataset ---
    st.header("📊 Existing Student Data")

    try:
        df = pd.read_csv("STUDENT BURNOUT LEVEL PREDICTION (Responses) - Form responses 1.csv")
        st.success("✅ Dataset loaded successfully!")
        st.dataframe(df)
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Please place your CSV file in the same folder as app.py.")
        df = None

    st.divider()

    # --- Predict for All Students ---
    if df is not None:
        st.subheader("🧮 Predict Burnout Levels for All Students")

        if st.button("🔍 Predict for All Students"):
            try:
                # Convert categorical data to match training
                df_encoded = pd.get_dummies(df, columns=["Study Hours", "Sleep Hours", "Stress"], drop_first=False)

                # Align feature names with the scaler
                for col in scaler.feature_names_in_:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0  # add missing columns

                df_encoded = df_encoded[scaler.feature_names_in_]

                # Scale and predict
                scaled = scaler.transform(df_encoded)
                predictions = model.predict(scaled)

                # Map numerical predictions to labels
                burnout_labels = {0: "Low", 1: "Medium", 2: "High"}
                df["Predicted Burnout Level"] = [burnout_labels[p] for p in predictions]

                st.success("✅ Burnout levels predicted successfully!")
                st.dataframe(df)

                # --- Show Chart ---
                st.subheader("📈 Burnout Level Distribution")

                counts = df["Predicted Burnout Level"].value_counts()

                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(counts)
                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
                    ax.set_title("Burnout Level Distribution")
                    st.pyplot(fig)

                # Option to download updated dataset
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Updated Dataset with Predictions",
                    data=csv,
                    file_name="student_burnout_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

    st.divider()

    # --- Predict New Burnout Level ---
    st.header("🧠 Predict a New Student's Burnout Level")

    study_hours = st.selectbox(
        "Study Hours per day",
        ["Less than 2 hours", "2-4 hours", "4-6 hours", "More than 6 hours"]
    )
    sleep_hours = st.selectbox(
        "Sleep Hours per day",
        ["Less than 4 hours", "4-6 hours", "6-8 hours", "More than 8 hours"]
    )
    stress = st.selectbox(
        "Stress Level",
        ["Low", "Moderate", "High"]
    )
    cgpa = st.number_input("CGPA (Out of 10)", min_value=0.0, max_value=10.0, step=0.1)
    attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, step=0.1)

    input_df = pd.DataFrame({
        "CGPA (Out of 10)": [cgpa],
        "Attendance Percentage": [attendance],
        "Study Hours_0": [1 if study_hours == "Less than 2 hours" else 0],
        "Study Hours_1": [1 if study_hours == "2-4 hours" else 0],
        "Study Hours_2": [1 if study_hours == "4-6 hours" else 0],
        "Sleep Hours_0": [1 if sleep_hours == "Less than 4 hours" else 0],
        "Sleep Hours_1": [1 if sleep_hours == "4-6 hours" else 0],
        "Sleep Hours_2": [1 if sleep_hours == "6-8 hours" else 0],
        "Stress_0": [1 if stress == "Low" else 0],
        "Stress_1": [1 if stress == "Moderate" else 0],
        "Stress_2": [1 if stress == "High" else 0],
    })

    # Scale features
    scaled_features = scaler.transform(input_df)

    # Predict
    if st.button("🎯 Predict Single Burnout Level"):
        prediction = model.predict(scaled_features)
        burnout_class = {0: "Low", 1: "Medium", 2: "High"}
        st.success(f"🔥 Predicted Burnout Level: {burnout_class[prediction[0]]}")

    # --- Optional: Summary ---
    if df is not None:
        st.divider()
        st.subheader("📊 Dataset Summary")
        st.write(df.describe())


# ------------------ PAGE 2: DATA REPORT ------------------
elif page == "📈 Data Report":
    st.title("📊 Student Data Report")

    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully!")

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        st.subheader("📈 Summary Statistics")
        st.write(df.describe())

        st.subheader("🎨 Visual Insights")

        # CGPA vs Stress
        if "Stress" in df.columns and "CGPA (Out of 10)" in df.columns:
            st.write("📊 Average CGPA by Stress Level")
            plt.figure(figsize=(8, 5))
            df.groupby("Stress")["CGPA (Out of 10)"].mean().plot(kind="bar", color="skyblue", edgecolor="black")
            plt.ylabel("Average CGPA")
            plt.xlabel("Stress Level")
            plt.title("CGPA vs Stress Level")
            st.pyplot(plt)

        # Attendance vs Burnout Level
        if "Attendance Percentage" in df.columns and "Burnout Level" in df.columns:
            st.write("📊 Average Attendance by Burnout Level")
            plt.figure(figsize=(8, 5))
            df.groupby("Burnout Level")["Attendance Percentage"].mean().plot(kind="bar", color="orange", edgecolor="black")
            plt.ylabel("Average Attendance")
            plt.xlabel("Burnout Level")
            plt.title("Attendance vs Burnout Level")
            st.pyplot(plt)
    else:
        st.info("👆 Upload a CSV file to generate insights.")

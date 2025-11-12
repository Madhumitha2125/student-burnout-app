import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🎓 Student Burnout Prediction Dashboard",
    page_icon="🔥",
    layout="wide",
)

# --- CUSTOM STYLES ---
st.markdown("""
    <style>
        .main {
            background-color: #F9FAFB;
        }
        div.block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        h1, h2, h3 {
            color: #1E3A8A;
        }
        .stButton>button {
            background-color: #1E40AF;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3B82F6;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- TITLE SECTION ---
st.title("🎓 Student Burnout Prediction Dashboard")
st.markdown("### Analyze, Predict, and Visualize Student Burnout Trends")

# Load Model & Scaler
model = joblib.load("burnout_model.joblib")
scaler = joblib.load("scaler.joblib")

# --- TABS LAYOUT ---
tabs = st.tabs(["📈 Dashboard", "🧠 Predict New Student", "📊 Dataset Insights"])

# ========================
# 📈 TAB 1: Dashboard View
# ========================
with tabs[0]:
    st.header("🔥 Predict Burnout for Uploaded Dataset")

    uploaded_file = st.file_uploader("Upload Student Dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully!")
        st.dataframe(df.head())

        try:
            df_encoded = pd.get_dummies(df, columns=["Study Hours", "Sleep Hours", "Stress"], drop_first=False)
            for col in scaler.feature_names_in_:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[scaler.feature_names_in_]

            scaled = scaler.transform(df_encoded)
            predictions = model.predict(scaled)
            burnout_labels = {0: "Low", 1: "Medium", 2: "High"}
            df["Predicted Burnout Level"] = [burnout_labels[p] for p in predictions]

            st.success("✅ Predictions completed successfully!")
            st.dataframe(df)

            # Burnout level distribution
            counts = df["Predicted Burnout Level"].value_counts()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔹 Burnout Level Distribution")
                st.bar_chart(counts)

            with col2:
                fig, ax = plt.subplots()
                ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=["#60A5FA", "#FACC15", "#F87171"])
                ax.set_title("Burnout Percentage Breakdown")
                st.pyplot(fig)

            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="student_burnout_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.info("📤 Please upload your dataset to begin.")

# ========================
# 🧠 TAB 2: Predict Single
# ========================
with tabs[1]:
    st.header("🧠 Predict a Single Student’s Burnout Level")

    col1, col2, col3 = st.columns(3)
    with col1:
        study_hours = st.selectbox("📘 Study Hours per day", ["Less than 2 hours", "2-4 hours", "4-6 hours", "More than 6 hours"])
        sleep_hours = st.selectbox("💤 Sleep Hours per day", ["Less than 4 hours", "4-6 hours", "6-8 hours", "More than 8 hours"])
    with col2:
        stress = st.selectbox("😣 Stress Level", ["Low", "Moderate", "High"])
        cgpa = st.number_input("🎯 CGPA (Out of 10)", min_value=0.0, max_value=10.0, step=0.1)
    with col3:
        attendance = st.number_input("🧾 Attendance (%)", min_value=0.0, max_value=100.0, step=0.1)

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

    scaled = scaler.transform(input_df)
    if st.button("🎯 Predict Burnout Level"):
        prediction = model.predict(scaled)
        burnout_class = {0: "Low", 1: "Medium", 2: "High"}
        st.success(f"🔥 Predicted Burnout Level: **{burnout_class[prediction[0]]}**")

# ========================
# 📊 TAB 3: Insights Page
# ========================
with tabs[2]:
    st.header("📊 Dataset Insights & Trends")
    st.markdown("Upload your CSV file to view data trends and summary statistics.")
    file = st.file_uploader("Upload CSV for Insights", type=["csv"], key="insights_upload")

    if file:
        data = pd.read_csv(file)
        st.success("✅ File uploaded!")
        st.write(data.describe())

        st.markdown("### 🔍 Correlation Heatmap (Numerical Features)")
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        if not numeric_data.empty:
            st.dataframe(numeric_data.corr())
        else:
            st.warning("No numeric data found.")
    else:
        st.info("Please upload a dataset to explore insights.")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed by Madhu Mitha 💻 | Powered by Streamlit | Enhanced Dashboard UI ✨")



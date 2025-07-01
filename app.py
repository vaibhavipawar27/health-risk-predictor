import streamlit as st
import pickle
import numpy as np
import pickle
import os

# Safer model loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error("❌ Could not load the model.")
    st.text(f"Error: {e}")
    st.stop()


# Load trained model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("🩺 Smart Health Risk Predictor")
st.subheader("🔍 Predict Your Risk for Diabetes")

# Sidebar for user input
st.sidebar.header("🧾 Patient Information")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 10, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

st.sidebar.header("📋 Health Metrics")

pregnancies = st.sidebar.slider("Pregnancies", 0, 15, 1)
glucose = st.sidebar.slider("Glucose Level", 50, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 40, 130, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 300, 79)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
age_input = st.sidebar.slider("Age (again for model)", 20, 80, age)

# Predict button
if st.sidebar.button("🔘 Predict Risk"):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age_input]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    st.success(f"👤 Name: {name}")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes")
    else:
        st.success(f"✅ Low Risk of Diabetes")

    st.metric(label="📊 Prediction Confidence", value=f"{prob:.2f}%")

    st.info("💡 Tips:")
    st.write("- Eat a balanced diet")
    st.write("- Exercise regularly")
    st.write("- Monitor blood sugar levels")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Vaibhavi Pawar")

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- Load and train model directly ---
df = pd.read_csv("bmi_categories_prediction.csv")

gender_encoder = LabelEncoder()
df['gender'] = gender_encoder.fit_transform(df['gender'])

target_encoder = LabelEncoder()
df['bmi_category'] = target_encoder.fit_transform(df['bmi_category'])

X = df[['height_cm', 'weight_kg', 'age', 'gender']]
y = df['bmi_category']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# --- Streamlit Interface ---
st.title("💪 BMI Category Predictor")
st.write("Predict your BMI category using your details!")

height = st.number_input("Height (cm)", 120.0, 220.0, 170.0)
weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
age = st.number_input("Age", 10, 100, 25)
gender = st.radio("Gender", ["Male", "Female"])

if st.button("Predict"):
    gender_val = gender_encoder.transform([gender])[0]
    input_df = pd.DataFrame({
        "height_cm": [height],
        "weight_kg": [weight],
        "age": [age],
        "gender": [gender_val]
    })

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    predicted_category = target_encoder.inverse_transform(prediction)[0]

    bmi_value = weight / ((height / 100) ** 2)

    st.success(f"📊 Your BMI: {bmi_value:.2f}")
    st.info(f"🏷️ Predicted BMI Category: **{predicted_category}**")

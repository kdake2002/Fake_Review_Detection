import streamlit as st
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🕵️ Fake Review Detector")

st.write("Enter a product or service review, and the model will predict if it's **genuine** or **fake**.")

# Input box
review_text = st.text_area("Enter Review:")

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("⚠️ Please enter some review text.")
    else:
        prediction = model.predict([review_text])[0]
        if prediction == 1:
            st.success("✅ Prediction: This review is **Genuine**.")
        else:
            st.error("❌ Prediction: This review is **Fake**.")

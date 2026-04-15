import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection App")

# Input
input_text = st.text_area("Enter News Article Text:")

if st.button("Check News"):
    if input_text:
        text_vector = vectorizer.transform([input_text])
        prediction = model.predict(text_vector)

        if prediction[0] == 1:
            st.success("✅ This is Real News")
        else:
            st.error("❌ This is Fake News")
    else:
        st.warning("Please enter some text")
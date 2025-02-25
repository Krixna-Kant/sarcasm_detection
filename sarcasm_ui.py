import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Model & Tokenizer
MODEL_PATH = "./saved_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Prediction Function
def predict_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = predictions[0][predicted_class].item()
    return "Sarcastic 😏" if predicted_class == 1 else "Not Sarcastic 🙂", confidence

# Streamlit UI
st.title("🔍 Multilingual Sarcasm Detector 😆🤖")
st.markdown("Detect sarcasm in multilingual social media posts (English, Hindi, Code-Mixed)")

user_input = st.text_area("📝 Enter a social media post:")

if st.button("Detect Sarcasm 🚀"):
    if user_input:
        result, confidence = predict_sarcasm(user_input)
        st.success(f"**Prediction:** {result}")
        st.info(f"**Confidence Score:** {confidence:.2%}")
    else:
        st.warning("⚠️ Please enter some text!")

st.markdown("---")
st.markdown("**Developed by Team Squirtle | Hackarena 🚀**")

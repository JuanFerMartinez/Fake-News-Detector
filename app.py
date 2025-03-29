# -----------------------------------------------------------------------------
# Project: Fake News Detection with TF-IDF and Naive Bayes
# Author: Juan Fernando Martínez Ruiz
# Year: 2025
# LinkedIn: https://www.linkedin.com/in/juanfermartinez/
#
# Copyright (c) 2025 Juan Fernando Martínez Ruiz
#
# This code is licensed under the MIT License.
# Unauthorized commercial use or reproduction without proper attribution is prohibited.
# -----------------------------------------------------------------------------


import streamlit as st
from src.cargar_datos import cargar_datos
from src.preprocesamiento import limpiar_texto
from src.modelo import entrenar_modelo, predecir_noticia, cargar_modelo_entrenado

# Configure the page
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("🧠 Fake News Detector")
st.write("This model uses **TF-IDF + Naive Bayes** to detect whether a news article is real or fake.")

# Load or train model with caching
@st.cache_resource(show_spinner=True)
def load_or_train_model():
    """
    Load model and vectorizer from disk, or train a new one if not found.
    Returns:
        tuple: model and vectorizer
    """
    model, vectorizer = cargar_modelo_entrenado()
    if model is None or vectorizer is None:
        st.warning("⚠️ Model not found. Training a new one...")
        data = cargar_datos()
        data = data.dropna(subset=['text'])
        data['clean_text'] = data['text'].apply(limpiar_texto)
        model, vectorizer = entrenar_modelo(data)
    return model, vectorizer

# Load model
with st.spinner("🔄 Loading model..."):
    model, vectorizer = load_or_train_model()

# Text input
user_input = st.text_area("✍️ Enter the news article text to analyze:")

# Prediction
if user_input:
    cleaned_text = limpiar_texto(user_input)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]

    st.markdown(f"##### 📝 Entered Text:\n> {user_input}")
    if prediction == 1:
        st.error("🔴 This article was classified as **FAKE**.")
    else:
        st.success("🟢 This article was classified as **REAL**.")

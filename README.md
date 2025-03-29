<div align="center">

# 🧠 Fake News Detector  
### Detecting Misinformation with Machine Learning + Streamlit  

![banner](https://img.shields.io/badge/NLP-TF--IDF-informational?style=flat-square&logo=python&color=4AB197)
![badge](https://img.shields.io/badge/Model-Naive%20Bayes-brightgreen?style=flat-square)
![status](https://img.shields.io/badge/Deployment-Streamlit-blueviolet?style=flat-square)
![license](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

</div>

---

## 📌 Overview

**Fake News Detector** is a web application that allows users to paste a news headline or body text and instantly determine whether it's **Real** or **Fake**.  
It uses **TF-IDF vectorization** and a **Multinomial Naive Bayes** model, trained on a public dataset of true and false news articles.

This project combines **Natural Language Processing (NLP)** with a user-friendly interface via **Streamlit**, making machine learning both interactive and accessible.

---

## 🎯 Use Cases

- 🔍 Detecting misinformation in news media  
- 🧪 NLP model experimentation  
- 🎓 Educational demo for ML/NLP techniques  
- 🧑‍💻 Portfolio project showcasing practical data science skills

---

## ✨ Features

✅ Clean, modern web interface  
✅ Real-time prediction (no page reloads)  
✅ Preprocessing: lowercasing, punctuation & stopword removal  
✅ Text vectorization with TF-IDF  
✅ Model training & evaluation with precision/recall metrics  
✅ Supports retraining and persistent models using `joblib`

---

## 🧪 Demo Screenshot

<img src="assets/screenshot.png" alt="Fake News Detector App Screenshot" width="700"/>

---

## 🛠️ Tech Stack

| Layer             | Technology                  |
|------------------|-----------------------------|
| **Frontend**     | Streamlit                   |
| **ML Model**     | Naive Bayes (MultinomialNB) |
| **Text Features**| TF-IDF                      |
| **NLP Preproc.** | NLTK                        |
| **Backend**      | Python 3.9+                 |
| **Persistence**  | Joblib                      |

---

## 📁 Project Structure

Modelonoticias/ │ ├── data/ # Raw datasets │ ├── Fake.csv │ └── True.csv │ ├── assets/ # Screenshots or media │ └── screenshot.png │ ├── src/ # Source code │ ├── cargar_datos.py # Load and merge datasets │ ├── eda.py # Exploratory Data Analysis │ ├── modelo.py # Model training and prediction │ ├── preprocesamiento.py # Text cleaning and preprocessing │ └── app.py # Streamlit UI app │ ├── main.py # Train + evaluate model from CLI ├── modelo.joblib # Saved Naive Bayes model ├── vectorizer.joblib # Saved TF-IDF vectorizer ├── requirements.txt └── README.md

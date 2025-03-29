import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocesamiento import limpiar_texto

# Paths to save the trained model and vectorizer
MODELO_PATH = "modelo.joblib"
VECTORIZADOR_PATH = "vectorizer.joblib"

def entrenar_modelo(data_limpio):
    """
    Train a Naive Bayes model using TF-IDF features from cleaned news text.

    Args:
        data_limpio (pd.DataFrame): DataFrame with a 'clean_text' column
                                    and a 'label' column (0 = real, 1 = fake).

    Returns:
        tuple: Trained model and fitted vectorizer.
    """
    # Remove empty texts
    data_limpio = data_limpio[data_limpio['clean_text'].str.strip().astype(bool)]

    # Split into features and labels
    X = data_limpio['clean_text']
    y = data_limpio['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes classifier
    modelo = MultinomialNB()
    modelo.fit(X_train_tfidf, y_train)

    # Predict and show metrics
    y_pred = modelo.predict(X_test_tfidf)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(modelo, MODELO_PATH)
    joblib.dump(vectorizer, VECTORIZADOR_PATH)

    return modelo, vectorizer

def cargar_modelo_entrenado():
    """
    Load a trained model and TF-IDF vectorizer from disk if they exist.

    Returns:
        tuple: Loaded model and vectorizer, or (None, None) if not found.
    """
    if os.path.exists(MODELO_PATH) and os.path.exists(VECTORIZADOR_PATH):
        modelo = joblib.load(MODELO_PATH)
        vectorizer = joblib.load(VECTORIZADOR_PATH)
        print("✅ Loaded model and vectorizer from disk.")
        return modelo, vectorizer
    else:
        print("⚠️ Model files not found.")
        return None, None

def predecir_noticia(texto, modelo, vectorizer):
    """
    Predict if a given news text is real or fake.

    Args:
        texto (str): Raw news text.
        modelo: Trained classifier.
        vectorizer: Fitted TF-IDF vectorizer.

    Returns:
        None. Prints the prediction result.
    """
    texto_limpio = limpiar_texto(texto)
    vector = vectorizer.transform([texto_limpio])
    pred = modelo.predict(vector)[0]

    print(f"\n📝 Input text:\n\"{texto}\"\n")
    if pred == 1:
        print("🔴 The news was classified as **FAKE**.")
    else:
        print("🟢 The news was classified as **REAL**.")

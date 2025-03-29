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

import re
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords the first time
nltk.download('stopwords')

# English stopword set
stop_words = set(stopwords.words('english'))

def limpiar_texto(texto):
    """
    Clean a given text by removing links, punctuation, digits, and stopwords.

    Args:
        texto (str): The input text to clean.

    Returns:
        str: Cleaned text.
    """
    texto = str(texto).lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)  # remove URLs
    texto = re.sub(r'<.*?>+', '', texto)  # remove HTML tags
    texto = re.sub(r'[%s]' % re.escape(string.punctuation), '', texto)  # remove punctuation
    texto = re.sub(r'\n', ' ', texto)  # remove line breaks
    texto = re.sub(r'\w*\d\w*', '', texto)  # remove words with numbers
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words]
    return " ".join(palabras)

def aplicar_limpieza(df):
    """
    Apply text cleaning to the 'text' column of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'text' column.

    Returns:
        pd.DataFrame: DataFrame with an additional 'clean_text' column.
    """
    df = df.copy()
    df['clean_text'] = df['text'].apply(limpiar_texto)
    return df

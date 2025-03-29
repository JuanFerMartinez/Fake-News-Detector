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
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cargar_datos import cargar_datos
from src.eda import analizar_distribucion
from src.preprocesamiento import aplicar_limpieza
from src.modelo import entrenar_modelo, predecir_noticia

def main():
    """
    Run the full pipeline: load data, explore, preprocess, train, and predict.
    """
    # Load and combine datasets
    print("📥 Loading dataset...")
    data = cargar_datos()

    # Show first rows
    print("📋 First rows of the combined dataset:")
    print(data.head())

    # Distribution of real vs. fake
    print("\n📊 Label distribution (0 = real, 1 = fake):")
    print(data['label'].value_counts())

    # Show distribution plot
    analizar_distribucion(data)

    # Clean the text
    print("\n🧹 Cleaning text...")
    data_limpio = aplicar_limpieza(data)

    # Text length stats
    print("\n🔍 Length of cleaned texts:")
    print(data_limpio['clean_text'].str.len().describe())

    # Show examples
    print("\n📌 Sample original vs cleaned text:")
    print(data_limpio[['text', 'clean_text']].head(2))

    # Train model
    print("\n🚀 Training model...")
    modelo, vectorizer = entrenar_modelo(data_limpio)

    # Test prediction
    test_text = "Donald Trump resigns as president of the United States"
    predecir_noticia(test_text, modelo, vectorizer)

if __name__ == '__main__':
    main()

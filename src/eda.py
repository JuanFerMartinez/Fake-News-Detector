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
import seaborn as sns
import matplotlib.pyplot as plt

def analizar_distribucion(data):
    """
    Display a bar plot showing the distribution of real and fake news.

    Args:
        data (pd.DataFrame): The dataset containing the 'label' column,
                             where 0 = real and 1 = fake.
    """
    sns.countplot(x='label', data=data)
    plt.title('News Distribution (0 = Real, 1 = Fake)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

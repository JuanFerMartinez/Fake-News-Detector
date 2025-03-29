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

import pandas as pd
import os

def cargar_datos():
    """
    Load and combine real and fake news datasets from the data directory.

    Returns:
        pd.DataFrame: Combined dataset with an added 'label' column
                      (0 = real, 1 = fake).
    """
    # Define the absolute path to the 'data' directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    # Load CSV files
    fake_path = os.path.join(data_dir, 'Fake.csv')
    true_path = os.path.join(data_dir, 'True.csv')

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Add label column: 1 for fake, 0 for real
    df_fake['label'] = 1
    df_true['label'] = 0

    # Combine datasets
    combined_df = pd.concat([df_fake, df_true], ignore_index=True)

    return combined_df

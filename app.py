import streamlit as st
import pandas as pd
from joblib import load
from preprocess import load_and_clean_data
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from io import StringIO

st.title("Détection de Fraude par Carte de Crédit")

# Charger le modèle
model = load('model/fraud_detection_model.joblib')

# Uploader un fichier CSV de nouvelles transactions
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        new_data = pd.read_csv(stringio)

        st.write("Aperçu des transactions :")
        st.write(new_data.head())

        # Nettoyage des données avant prédiction
        new_data_clean = load_and_clean_data(new_data)

        imputer = SimpleImputer(strategy='mean')  # Remplacer les NaN par la moyenne
        new_data_clean = imputer.fit_transform(new_data_clean)

        # Prédire la fraude
        predictions = model.predict(new_data_clean)

        new_data['Fraud_Prediction'] = predictions

        # Ajouter la colonne explicite
        new_data['Prediction_Explicit'] = new_data['Fraud_Prediction'].apply(lambda x: 'Fraude' if x == 1 else 'Non-Fraude')

        st.write("Résultats de la prédiction :")
        st.write(new_data[['Time', 'Amount', 'Fraud_Prediction', 'Prediction_Explicit']])


    except Exception as e:
        st.error(f"Erreur : {e}")

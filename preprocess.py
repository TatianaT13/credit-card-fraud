# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(df):
    # Supposer que 'df' est déjà un DataFrame chargé
    # Normaliser les colonnes 'Amount' et 'Time'
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    
    # Supprimer les colonnes originales 'Amount' et 'Time'
    df = df.drop(columns=['Amount', 'Time'])

    return df

import pandas as pd
from joblib import load
from preprocess import load_and_clean_data
from sklearn.impute import SimpleImputer

# Charger le modèle
model = load('model/fraud_detection_model.joblib')

# Charger et prétraiter les nouvelles transactions
new_data = pd.read_csv('data/new_transactions.csv')

# Prétraiter les données (normalisation, etc.)
new_data_clean = load_and_clean_data(new_data)

# Ajouter un imputer pour gérer les valeurs NaN
imputer = SimpleImputer(strategy='mean')  # Remplacer les NaN par la moyenne
new_data_clean = imputer.fit_transform(new_data_clean)

# Restaurer les noms de colonnes après l'imputation
new_data_clean = pd.DataFrame(new_data_clean, columns=new_data.columns.drop(['Amount', 'Time']))

# Prédire la fraude
predictions = model.predict(new_data_clean)

# Ajouter les prédictions au DataFrame original
new_data['Fraud_Prediction'] = predictions

# Afficher les résultats avec des colonnes existantes
print("Résultats de la prédiction :")
print(new_data[['Time', 'Amount', 'Fraud_Prediction']])

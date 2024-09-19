# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from preprocess import load_and_clean_data

# Charger et prétraiter les données
df = load_and_clean_data('data/creditcard.csv')

# Diviser les données en variables explicatives et cible
X = df.drop(columns=['Class'])  # 'Class' est la colonne cible
y = df['Class']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=['Non-Fraude', 'Fraude']))

# Sauvegarder le modèle pour une utilisation future
dump(model, 'model/fraud_detection_model.joblib')
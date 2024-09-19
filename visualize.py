# visualize.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import load_and_clean_data

# Charger et prétraiter les données
df = load_and_clean_data('data/creditcard.csv')

# Visualisation de la distribution des classes
sns.countplot(x='Class', data=df)
plt.title("Distribution des Transactions (Fraude vs Non-Fraude)")
plt.show()

# Visualisation de la corrélation entre les features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Corrélation entre les Variables")
plt.show()
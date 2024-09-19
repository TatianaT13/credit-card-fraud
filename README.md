# Détection de Fraude par Carte de Crédit avec Machine Learning

Ce projet vise à détecter les transactions frauduleuses à l'aide de modèles de machine learning. Le modèle est entraîné à partir du dataset [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) et est capable de prédire si une transaction est frauduleuse ou non. Le projet inclut également une interface interactive pour faire des prédictions sur de nouvelles transactions.

## Objectifs

- Prétraiter les données de transactions.
- Entraîner un modèle de machine learning pour prédire les fraudes.
- Évaluer les performances du modèle à l'aide de métriques comme l'accuracy, le rappel, et le F1-score.
- Permettre la prédiction de fraudes sur des transactions en temps réel ou des données historiques via une interface.

## Installation

1. **Cloner le dépôt :**

```bash
   git clone https://github.com/TatianaT13/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
```

Créer un environnement virtuel et activer-le :

```bash
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
```

## Installer les dépendances :


```bash
pip install -r requirements.txt
```
## Télécharger le dataset :
Télécharge le dataset Credit Card Fraud Detection depuis Kaggle.
Place le fichier creditcard.csv dans le dossier data/.

# Utilisation
Entraîner le Modèle
Pour entraîner le modèle de détection de fraude, exécute le script train.py :

```bash
python train.py
```
Cela entraînera le modèle sur le dataset et l'enregistrera sous le nom fraud_detection_model.joblib dans le dossier model/.

Faire des Prédictions sur de Nouvelles Transactions
Pour prédire si de nouvelles transactions sont frauduleuses, exécute le script predict.py après avoir préparé un fichier CSV de nouvelles transactions dans le dossier data/ :

```bash
python predict.py
```

Le fichier predict.py chargera les nouvelles transactions, appliquera le modèle de détection de fraude, et affichera les résultats dans le terminal.


Exécute le script app.py :

```bash
streamlit run app.py
```

Télécharge un fichier CSV contenant les nouvelles transactions et l'application affichera les résultats des prédictions de fraude.

Le modèle est évalué à l'aide des métriques suivantes :

Accuracy : Pourcentage des prédictions correctes (fraude et non-fraude).
Rappel : Capacité à identifier correctement les transactions frauduleuses.
F1-score : Moyenne harmonique entre la précision et le rappel.
Exemples de Résultats :
Après l'entraînement, les performances du modèle seront affichées, y compris des détails comme :

Accuracy: 0.9993
              précision  rappel    f1-score   support

   Non-Fraude     1.00     1.00      1.00     56863
       Fraude     0.87     0.77      0.82       99

    accuracy                         1.00     56962
   macro avg     0.93     0.89      0.91     56962
weighted avg     1.00     1.00      1.00     56962

## Améliorations Possibles

Optimisation des Hyperparamètres : Utiliser GridSearchCV pour ajuster les paramètres du modèle.
Gestion des Données Déséquilibrées : Expérimenter avec des techniques de suréchantillonnage comme SMOTE ou le sous-échantillonnage des classes majoritaires.
Ajout de Modèles : Tester d'autres modèles de classification (SVM, réseaux neuronaux) et comparer les performances.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# 1. Caricamento del dataset generato precedentemente
try:
    df = pd.read_csv('dataset_hotel.csv')
    print("Dataset caricato con successo!")
except FileNotFoundError:
    print("Errore: Non trovo dataset_hotel.csv. Esegui prima dataset_generator.py")
    exit()

# 2. Definizione delle variabili
# X è il testo della recensione, y sono le etichette che vogliamo prevedere
X = df['testo_recensione']
y_reparto = df['reparto']
y_sentiment = df['sentiment']

# 3. Creazione della Pipeline di Machine Learning
# Usiamo TF-IDF per trasformare le parole in numeri e Logistic Regression per classificare
def create_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression())
    ])

# Creiamo due modelli separati: uno per il reparto e uno per il sentiment
model_reparto = create_pipeline()
model_sentiment = create_pipeline()

# 4. Addestramento dei modelli
print("Addestramento dei modelli in corso...")
model_reparto.fit(X, y_reparto)
model_sentiment.fit(X, y_sentiment)

# 5. Salvataggio del "cervello" del progetto
# Salviamo tutto in un file .pkl che verrà usato dalla Web App
modelli_finiti = {
    'reparto': model_reparto,
    'sentiment': model_sentiment
}

joblib.dump(modelli_finiti, 'modello_hotel.pkl')
print("✅ Modello addestrato e salvato come 'modello_hotel.pkl'")
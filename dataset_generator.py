import pandas as pd
import random

# 1. Definizione dei cluster operativi (Dipartimenti)
aree_operative = ['Housekeeping', 'Reception', 'F&B']

# 2. NUOVO DIZIONARIO OTTIMIZZATO (Inseriscilo qui)
frasi_campione = {
    'Housekeeping': [
        "Ambiente igienizzato e profumato", "Arredi curati e pulizia profonda", 
        "Stanza in condizioni impeccabili", "Lenzuola pulitissime e camera ordinata",
        "Polvere sotto il letto e lenzuola sporche", "Bagno non pulito",
        "Asciugamani mancanti", "Tutto splende, pulizia eccellente"
    ],
    'Reception': [
        "Check-in rapido e cortesia estrema", "Staff professionale", 
        "Accoglienza calorosa", "Attesa infinita al check-in",
        "Personale scortese", "Problemi con la prenotazione",
        "Receptionist molto gentile e disponibile"
    ],
    'F&B': [
        "Esperienza gastronomica di livello", "Materie prime eccellenti", 
        "Cibo delizioso e servizio rapido", "Colazione scarsa",
        "Cibo freddo e cameriere sgarbato", "Ristorante disorganizzato",
        "Cena fantastica, tutto molto buono"
    ]
}

def generate_dataset(num_rows=1500):
    data = []
    # Parole chiave che indicano sicuramente un sentimento negativo
    parole_negative = ["sporche", "non pulito", "mancanti", "disordinata", "attesa", 
                       "scortese", "problemi", "scarsa", "freddo", "sgarbato", 
                       "disorganizzato", "polvere"]
    
    for _ in range(num_rows):
        reparto = random.choice(aree_operative)
        recensione = random.choice(frasi_campione[reparto])
        
        # Logica migliorata: se la frase contiene una parola negativa, il sentiment è Negativo
        if any(parola in recensione.lower() for parola in parole_negative):
            sentiment = "Negativo"
        else:
            sentiment = "Positivo"
            
        data.append([recensione, reparto, sentiment])
    
    df = pd.DataFrame(data, columns=['testo_recensione', 'reparto', 'sentiment'])
    df.to_csv('dataset_hotel.csv', index=False, encoding='utf-8')
    print(f"✅ Dataset aggiornato con {num_rows} righe.")

if __name__ == "__main__":
    generate_dataset()
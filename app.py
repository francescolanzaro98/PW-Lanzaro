import streamlit as st
import joblib

# Configurazione pagina
st.set_page_config(page_title="Hospitality AI - Analisi Recensioni", page_icon=" ")

# 1. Caricamento del modello salvato
@st.cache_resource # Serve per caricare il modello una sola volta e velocizzare l'app
def load_models():
    return joblib.load('modello_hotel.pkl')

try:
    models = load_models()
    model_reparto = models['reparto']
    model_sentiment = models['sentiment']
except:
    st.error("Errore: Il file 'modello_hotel.pkl' non è stato trovato. Esegui prima 'train_model.py'")
    st.stop()

# 2. Interfaccia Utente
st.title("Hospitality Intelligence")
st.subheader("Classificazione Automatica e Analisi del Sentiment")
st.write("Inserisci una recensione dell'hotel per smistarla al reparto corretto.")

# Area di testo per l'input dell'utente
user_input = st.text_area("Testo della recensione:", placeholder="Esempio: La camera era pulitissima ma la colazione era fredda...")

if st.button("Analizza Recensione"):
    if user_input:
        # 3. Predizione
        pred_reparto = model_reparto.predict([user_input])[0]
        pred_sentiment = model_sentiment.predict([user_input])[0]
        
        # 4. Visualizzazione Risultati
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Reparto Destinatario", value=pred_reparto)
        
        with col2:
            color = "green" if pred_sentiment == "Positivo" else "red"
            st.markdown(f"Sentiment: **:{color}[{pred_sentiment}]**")
            
        st.info(f"Logica: La recensione è stata inoltrata automaticamente al team di **{pred_reparto}**.")
    else:
        st.warning("Per favore, inserisci un testo per procedere.")

# Footer informativo per la tesi
st.sidebar.markdown(" Project Work : Automazione e ottimizzazione dei flussi operativi nel settore Hospitality: un approccio basato su Machine Learning per l'analisi delle recensioni.")
st.sidebar.info("Sviluppato da: **Francesco Lanzaro**")
st.sidebar.write("Modello: Logistic Regression")

st.sidebar.write("Vettorizzazione: TF-IDF")




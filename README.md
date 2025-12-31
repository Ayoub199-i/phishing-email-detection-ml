# Phishing Email Detection with Machine Learning

# Descrizione
Questo progetto utilizza il **Machine Learning** per rilevare email di phishing.  
Il modello analizza il testo delle email e le classifica come **Phishing** o **Legittime (Safe)**.

L'obiettivo è combinare **ML e cybersecurity** per dimostrare come un sistema automatico può identificare email sospette.

---

##Struttura del progetto

phishing_ml_project/
├─ data/
│ ├─ Phishing_validation_emails.csv
├─ phishingemail.ipynb
├─ venv/ (ambiente virtuale)
├─ .gitignore

yaml
Copia codice

- `data/` → contiene il dataset di email phishing e legittime  
- `phishingemail.ipynb` → notebook con tutto il codice per preprocessing, TF-IDF, train/test split e modello ML  
- `venv/` → ambiente virtuale Python (non caricato su GitHub)  
- `.gitignore` → ignora venv e file temporanei  

---

##  Tecnologie e librerie
- Python 3.11
- Jupyter Notebook
- Pandas
- scikit-learn
- matplotlib / seaborn (opzionali per visualizzazione)
- re (regex per pulizia del testo)

---

##  Funzionamento
1. Carica il dataset:
```python
import pandas as pd
df = pd.read_csv('data/Phishing_validation_emails.csv')
Pulizia del testo:

Minuscolo

Rimozione punteggiatura e numeri

Rimozione URL

Rimozione spazi extra

TF-IDF Vectorization:

python
Copia codice
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['clean_text'])
Train/Test split:

python
Copia codice
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['Email Type'], test_size=0.2, random_state=42, stratify=df['Email Type'])
Modello ML — Logistic Regression:

python
Copia codice
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
Valutazione del modello:

python
Copia codice
from sklearn.metrics import classification_report, accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
Risultati
Dataset: 2000 email (Phishing + Legittime)

Accuracy sul test set: 1.0 (dataset piccolo / pulito)

Precision e recall su phishing molto alte

Nota: il dataset è piccolo, quindi i risultati potrebbero essere ottimistici. Per un modello più robusto è consigliabile usare dataset più grandi e diversificati.

 Testare nuove email
È possibile testare email reali con la funzione:

python

def predict_email(text):
    text_clean = clean_text(text)
    vector = vectorizer.transform([text_clean])
    prediction = model.predict(vector)[0]
    return prediction
Esempio:

python

email_test = "Dear user, click here to verify your account..."
print(predict_email(email_test))
Collegamenti
Dataset di esempio: Phishing_validation_emails.csv

GitHub Repository: https://github.com/Ayoub199-i/phishing-email-detection-ml

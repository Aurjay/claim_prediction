Dieses Projekt ist eine Machine-Learning-Anwendung, die Vorhersagen darüber trifft, ob ein Versicherungsanspruch geltend gemacht wird. Es verwendet einen trainierten XGBoost-Klassifikator und bietet eine benutzerfreundliche Schnittstelle über eine Streamlit-App.

![image](https://github.com/user-attachments/assets/e2c2f645-ed78-474f-a8f8-77826c6eaf3b)

---

## Funktionen

- **Datenvorverarbeitung**: Automatische Handhabung fehlender Werte und Skalierung der Eingabedaten.
- **Modellentwicklung**: Ein XGBoost-Klassifikator wird mit ausgewählten Funktionen trainiert.
- **Web-App**: Eine Streamlit-basierte Benutzeroberfläche zur Vorhersage von Versicherungsansprüchen.
- **Interaktivität**: Benutzer können Eingabewerte für Schlüsselmerkmale bereitstellen, und die App gibt eine Vorhersage zurück.

## Dataset
[Car Insurance Claim Prediction - Classification](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification)

## Technologie-Stack

- **Python**: Primäre Programmiersprache
- **Pandas**: Datenverarbeitung
- **XGBoost**: Modelltraining
- **Streamlit**: Web-App-Framework
- **Joblib**: Speichern und Laden des trainierten Modells
- **Scikit-learn**: Skalierung der Daten und Modellauswertung

---

## Anforderungen

Installieren Sie die folgenden Python-Pakete:

```bash
pip install pandas numpy joblib scikit-learn xgboost streamlit
```

---

## Anleitung zur Nutzung

### 1. Datenverarbeitung und Modellentwicklung

- Platzieren Sie die Datei `train.csv` im Projektverzeichnis.
- Führen Sie den folgenden Code aus, um das Modell zu trainieren und zu speichern:

```bash
python train_model.py
```

---

### 2. Starten der Streamlit-App

- Stellen Sie sicher, dass die Datei `xgb_model.joblib` im selben Verzeichnis wie `app.py` vorhanden ist.
- Starten Sie die Streamlit-App mit dem folgenden Befehl:

```bash
streamlit run app.py
```

- Öffnen Sie die angegebene URL (standardmäßig [http://localhost:8501](http://localhost:8501)) in Ihrem Browser.

---

## Merkmale für die Vorhersage

| Feature                 | Beschreibung                                   | Bereich       |
|-------------------------|-----------------------------------------------|---------------|
| `policy_tenure`         | Laufzeit der Police (in Jahren)               | 0 bis 50      |
| `age_of_policyholder`   | Alter des Versicherungsnehmers (in Jahren)    | 18 bis 100    |
| `is_adjustable_steering`| Verstellbare Lenkung (0 = Nein, 1 = Ja)       | 0 oder 1      |
| `cylinder`              | Anzahl der Zylinder                           | 1 bis 12      |

---

## Vorhersageergebnisse

- **1 (Ja)**: Der Benutzer wird höchstwahrscheinlich einen Versicherungsanspruch geltend machen.
- **0 (Nein)**: Der Benutzer wird höchstwahrscheinlich keinen Versicherungsanspruch geltend machen.

---

## Projektstruktur

```
.
├── train.csv               # Datensatz für das Modelltraining
├── train_model.py          # Skript zur Datenvorbereitung und Modellentwicklung
├── app.py                  # Streamlit-App für Vorhersagen
├── xgb_model.joblib        # Gespeichertes XGBoost-Modell
├── README.md               # Projektdokumentation
```

---

## Autor

Erstellt von **Deepak Raj**.

- **Website**: [https://www.deepakraj.site/](https://www.deepakraj.site/)
- **GitHub**: [https://github.com/Aurjay/claim_prediction/tree/main](https://github.com/Aurjay/claim_prediction/tree/main)

---

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Weitere Informationen finden Sie in der Datei `LICENSE`.

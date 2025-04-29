# 🎤 Reconnaissance des Émotions Vocales

Ce projet implémente un système de reconnaissance des émotions à partir de la voix en utilisant le jeu de données TESS et un modèle Keras.

## 📁 Structure du projet

- `app.py` : Script principal pour entraîner et évaluer le modèle.
- `analyse.ipynb` : Notebook Jupyter pour l'analyse exploratoire des données et l'entraînement du modèle.
- `model.keras` : Modèle entraîné sauvegardé au format Keras.
- `TESS Toronto emotional speech set data/` : Dossier contenant le jeu de données TESS.
- `Reconnaissance vocale des émotions.pdf` : Rapport détaillé du projet.

## 🗂️ Jeu de données

Le projet utilise le jeu de données [TESS (Toronto Emotional Speech Set)](https://tspace.library.utoronto.ca/handle/1807/24487), qui contient des enregistrements vocaux simulant différentes émotions.

Les émotions incluses sont :
- Bonheur
- Tristesse
- Colère
- Peur
- Surprise
- Neutre

## 🛠️ Installation

1. **Cloner le dépôt :**

   ```bash
   git clone https://github.com/Houssam-Ibnchakroune/emotion-voice-recognition.git
   cd emotion-voice-recognition
2. **Installer les dépendances :**

   ```bash
   pip install -r requirements.txt
## ▶️ Utilisation
  ouvrez le terminal de le projet puis lancer la commande :

```bash
   streamlit run app.pyy 
```


## 📊 Résultats

Le modèle atteint une précision de 92% sur le jeu de test.


## 📄 Rapport

Pour plus de détails sur la méthodologie, les résultats et les conclusions, veuillez consulter le fichier Reconnaissance vocale des émotions.pdf.


## 🤝 Contribution
Les contributions sont les bienvenues ! N'hésitez pas à forker le projet et à proposer des améliorations via des pull requests.

Projet réalisé par Houssam Ibnchakroune.

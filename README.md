# 🚫 Détection de Spam SMS

Projet de classification de messages SMS en spam ou non-spam (ham) utilisant le machine learning.

## 📋 Description

Ce projet utilise un modèle de **classification supervisée** pour détecter automatiquement les SMS de spam. Le système combine :
- **Vectorisation TF-IDF** : pour transformer les textes en nombres
- **Naive Bayes** : algorithme de classification simple mais efficace pour le texte

## 🛠️ Technologies utilisées

- **Python 3.8+**
- **Pandas** : manipulation des données
- **Scikit-learn** : machine learning (TF-IDF, Naive Bayes)
- **Pickle** : sauvegarde du modèle

## 📁 Structure du projet

```
detection-spam-sms/
│
├── src/
│   ├── data_preparation.py    # Nettoyage et préparation des données
│   ├── train_model.py          # Entraînement du modèle
│   └── predict.py              # Prédictions sur nouveaux messages
│
├── data/
│   └── sms_spam.csv            # Dataset (à ajouter)
│
├── models/
│   ├── spam_detector.pkl       # Modèle entraîné (généré)
│   └── vectorizer.pkl          # Vectoriseur TF-IDF (généré)
│
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 🚀 Installation

1. Cloner le repository :
```bash
git clone https://github.com/ilyes-elhamdi/detection-spam-sms.git
cd detection-spam-sms
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Le projet inclut déjà :
   - Un fichier d'exemple (`data/example_sms_spam.csv`) pour tester rapidement
   - Un modèle pré-entraîné sur 5,569 messages réels (à télécharger si disponible)
   
   Pour entraîner avec vos propres données :
   - Format attendu : fichier TSV avec colonnes `label` (spam/ham) et `message`

## 💻 Utilisation

### 1. Entraîner le modèle

```bash
cd src
python train_model.py
```

Cela va :
- Nettoyer les données
- Entraîner le modèle
- Afficher les performances
- Sauvegarder le modèle dans `models/`

### 2. Faire des prédictions

Mode interactif :
```bash
cd src
python predict.py
```

Dans le code Python :
```python
from predict import load_model, predict_message

# Charger le modèle
model, vectorizer = load_model()

# Prédire un message
message = "Congratulations! You've won a free iPhone. Click here to claim!"
label, confidence = predict_message(message, model, vectorizer)

print(f"Résultat: {label} (confiance: {confidence:.2f}%)")
```

## 📊 Résultats obtenus

Le modèle a été entraîné sur **5,569 messages SMS réels** et atteint une exactitude de **97.13%**.

### Performances détaillées :
```
✓ Exactitude: 97.13%

Rapport de classification:
              precision    recall  f1-score   support
Ham              0.97      1.00      0.98       965
Spam             1.00      0.79      0.88       149

Matrice de confusion:
  - Vrais négatifs (Ham correct): 965
  - Faux positifs (Ham prédit Spam): 0
  - Faux négatifs (Spam prédit Ham): 32
  - Vrais positifs (Spam correct): 117
```

### 🎯 Points forts :
- **100% de précision** sur la détection de spam (pas de faux positifs)
- **97% de précision** sur les messages normaux
- **Aucun message normal** classé comme spam par erreur
- Modèle entraîné sur données réelles (UCI ML Repository)

## 🔧 Fonctionnalités

- ✅ Nettoyage automatique des messages (URLs, numéros, caractères spéciaux)
- ✅ Vectorisation TF-IDF avec bigrammes
- ✅ Classification Naive Bayes
- ✅ Mode prédiction interactif
- ✅ Sauvegarde/chargement du modèle


## 👤 Auteur

**Ilyes Elhamdi**
- LinkedIn: [ilyes-elhamdi](https://www.linkedin.com/in/ilyes-elhamdi-320202248)
- Email: ilyeshamdi48@gmail.com

## 📄 Licence

Projet personnel - libre d'utilisation à des fins éducatives

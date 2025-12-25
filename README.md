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

3. Télécharger un dataset de SMS :
   - Dataset recommandé : [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
   - Placer le fichier dans `data/sms_spam.csv`
   - Format attendu : colonnes `label` (spam/ham) et `message`

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

## 📊 Résultats attendus

Le modèle devrait atteindre une exactitude d'environ **97-98%** sur le dataset SMS Spam Collection.

Exemple de résultats :
```
✓ Exactitude: 97.85%

Rapport de classification:
              precision    recall  f1-score
Ham              0.99      0.99      0.99
Spam             0.95      0.94      0.94
```

## 🔧 Fonctionnalités

- ✅ Nettoyage automatique des messages (URLs, numéros, caractères spéciaux)
- ✅ Vectorisation TF-IDF avec bigrammes
- ✅ Classification Naive Bayes
- ✅ Mode prédiction interactif
- ✅ Sauvegarde/chargement du modèle

## 📝 Améliorations possibles

- [ ] Interface web avec Flask
- [ ] Support de plusieurs langues
- [ ] Tester d'autres algorithmes (SVM, Random Forest)
- [ ] API REST pour intégration
- [ ] Dashboard de monitoring

## 👤 Auteur

**Ilyes Elhamdi**
- LinkedIn: [ilyes-elhamdi](https://www.linkedin.com/in/ilyes-elhamdi-320202248)
- Email: ilyeshamdi48@gmail.com

## 📄 Licence

Projet personnel - libre d'utilisation à des fins éducatives

"""
Script pour entraîner le modèle de détection de spam
Utilise la vectorisation TF-IDF et un classifieur Naive Bayes
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preparation import load_data, prepare_dataset, split_data


def create_vectorizer(max_features=3000):
    """
    Crée un vectoriseur TF-IDF pour transformer les textes en nombres
    max_features: nombre maximum de mots à garder
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',  # Enlever les mots courants comme "the", "is", etc.
        ngram_range=(1, 2)     # Utiliser les mots seuls et les paires de mots
    )
    return vectorizer


def train_model(X_train, y_train):
    """
    Entraîne un modèle Naive Bayes sur les données d'entraînement
    Ce modèle est simple mais efficace pour la classification de texte
    """
    print("\nEntraînement du modèle...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("✓ Modèle entraîné")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle sur l'ensemble de test
    """
    print("\nÉvaluation du modèle...")
    
    # Faire des prédictions
    y_pred = model.predict(X_test)
    
    # Calculer l'exactitude
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Exactitude: {accuracy * 100:.2f}%")
    
    # Afficher le rapport détaillé
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion:")
    print(f"  Vrais négatifs (Ham correct): {cm[0][0]}")
    print(f"  Faux positifs (Ham prédit Spam): {cm[0][1]}")
    print(f"  Faux négatifs (Spam prédit Ham): {cm[1][0]}")
    print(f"  Vrais positifs (Spam correct): {cm[1][1]}")
    
    return accuracy


def save_model(model, vectorizer, model_path='../models/spam_detector.pkl', 
               vectorizer_path='../models/vectorizer.pkl'):
    """
    Sauvegarde le modèle et le vectoriseur pour une utilisation future
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\n✓ Modèle sauvegardé dans '{model_path}'")
    print(f"✓ Vectoriseur sauvegardé dans '{vectorizer_path}'")


# Pipeline complète d'entraînement
if __name__ == "__main__":
    print("=" * 50)
    print("ENTRAÎNEMENT DU DÉTECTEUR DE SPAM SMS")
    print("=" * 50)
    
    # 1. Charger et préparer les données
    df = load_data('../data/sms_spam.csv')
    
    if df is not None:
        df_clean = prepare_dataset(df)
        
        if df_clean is not None:
            # 2. Diviser les données
            X_train, X_test, y_train, y_test = split_data(df_clean)
            
            # 3. Créer et entraîner le vectoriseur
            print("\nVectorisation des textes...")
            vectorizer = create_vectorizer()
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            print(f"✓ Textes transformés en {X_train_vec.shape[1]} features")
            
            # 4. Entraîner le modèle
            model = train_model(X_train_vec, y_train)
            
            # 5. Évaluer le modèle
            accuracy = evaluate_model(model, X_test_vec, y_test)
            
            # 6. Sauvegarder le modèle
            save_model(model, vectorizer)
            
            print("\n" + "=" * 50)
            print("✓ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
            print("=" * 50)

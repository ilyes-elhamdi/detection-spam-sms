"""
Script pour prédire si un SMS est un spam ou non
Utilise le modèle déjà entraîné
"""

import pickle
from data_preparation import clean_text


def load_model(model_path='../models/spam_detector.pkl', 
               vectorizer_path='../models/vectorizer.pkl'):
    """
    Charge le modèle et le vectoriseur sauvegardés
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("✓ Modèle chargé avec succès")
        return model, vectorizer
    
    except FileNotFoundError:
        print("✗ Modèle non trouvé. Exécutez d'abord 'train_model.py'")
        return None, None


def predict_message(message, model, vectorizer):
    """
    Prédit si un message est un spam ou non
    Retourne la prédiction et la probabilité
    """
    # Nettoyer le message
    clean_msg = clean_text(message)
    
    # Transformer en vecteur
    message_vec = vectorizer.transform([clean_msg])
    
    # Faire la prédiction
    prediction = model.predict(message_vec)[0]
    
    # Obtenir la probabilité
    proba = model.predict_proba(message_vec)[0]
    
    # Interpréter le résultat
    label = "SPAM" if prediction == 1 else "HAM (normal)"
    confidence = proba[prediction] * 100
    
    return label, confidence


def predict_batch(messages, model, vectorizer):
    """
    Prédit plusieurs messages en une fois
    """
    results = []
    
    for msg in messages:
        label, confidence = predict_message(msg, model, vectorizer)
        results.append({
            'message': msg,
            'prediction': label,
            'confidence': f"{confidence:.2f}%"
        })
    
    return results


# Interface en ligne de commande
if __name__ == "__main__":
    print("=" * 60)
    print("DÉTECTEUR DE SPAM SMS")
    print("=" * 60)
    
    # Charger le modèle
    model, vectorizer = load_model()
    
    if model is not None and vectorizer is not None:
        print("\n[Mode interactif] Tapez vos messages pour les analyser")
        print("Tapez 'quit' ou 'exit' pour quitter\n")
        
        # Boucle interactive
        while True:
            message = input("Message à analyser: ")
            
            if message.lower() in ['quit', 'exit', 'q']:
                print("\nAu revoir!")
                break
            
            if message.strip() == '':
                continue
            
            # Faire la prédiction
            label, confidence = predict_message(message, model, vectorizer)
            
            # Afficher le résultat
            print(f"\n→ Résultat: {label}")
            print(f"→ Confiance: {confidence:.2f}%\n")
            print("-" * 60)

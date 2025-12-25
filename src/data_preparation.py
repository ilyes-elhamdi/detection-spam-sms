"""
Script pour préparer et nettoyer les données SMS
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Charge le fichier CSV contenant les SMS
    Le format attendu: colonne 'label' (spam/ham) et colonne 'message'
    """
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
        print(f"✓ Données chargées: {len(df)} messages")
        return df
    except Exception as e:
        print(f"✗ Erreur lors du chargement: {e}")
        return None


def clean_text(text):
    """
    Nettoie le texte en enlevant les caractères spéciaux et en passant en minuscules
    Garde uniquement les lettres et les espaces
    """
    # Mettre en minuscule
    text = text.lower()
    
    # Enlever les URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Enlever les numéros de téléphone (format simple)
    text = re.sub(r'\d{3,}', '', text)
    
    # Garder seulement les lettres et espaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Enlever les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def prepare_dataset(df):
    """
    Prépare le dataset en nettoyant les messages et en créant les labels numériques
    """
    # Vérifier que les colonnes existent
    if 'label' not in df.columns or 'message' not in df.columns:
        print("✗ Les colonnes 'label' et 'message' sont requises")
        return None
    
    # Nettoyer tous les messages
    print("Nettoyage des messages...")
    df['clean_message'] = df['message'].apply(clean_text)
    
    # Convertir les labels en nombres (spam = 1, ham = 0)
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # Enlever les lignes vides après nettoyage
    df = df[df['clean_message'].str.len() > 0]
    
    print(f"✓ Dataset préparé: {len(df)} messages valides")
    print(f"  - Spam: {sum(df['label_num'] == 1)}")
    print(f"  - Ham (non-spam): {sum(df['label_num'] == 0)}")
    
    return df


def split_data(df, test_size=0.2):
    """
    Divise les données en ensemble d'entraînement et de test
    """
    X = df['clean_message']
    y = df['label_num']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n✓ Données divisées:")
    print(f"  - Entraînement: {len(X_train)} messages")
    print(f"  - Test: {len(X_test)} messages")
    
    return X_train, X_test, y_train, y_test


# Exemple d'utilisation si on exécute ce script directement
if __name__ == "__main__":
    # Charger les données
    df = load_data('../data/sms_spam.csv')
    
    if df is not None:
        # Préparer le dataset
        df_clean = prepare_dataset(df)
        
        if df_clean is not None:
            # Diviser les données
            X_train, X_test, y_train, y_test = split_data(df_clean)
            
            # Sauvegarder les données nettoyées
            df_clean.to_csv('../data/sms_spam_clean.csv', index=False)
            print("\n✓ Données nettoyées sauvegardées dans 'data/sms_spam_clean.csv'")

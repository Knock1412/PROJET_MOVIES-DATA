# src/data/make_dataset.py
import pandas as pd
import numpy as np
import os
import pickle
from scipy import sparse
from scipy.sparse import csr_matrix

def load_data(raw_data_path='data/raw'):
    """
    Charge les fichiers rating.csv et movie.csv depuis le dossier raw.
    Retourne un DataFrame fusionné initial.
    """
    print("--- [Data] Chargement des données brutes ---")
    
    # Construction des chemins
    ratings_path = os.path.join(raw_data_path, 'rating.csv')
    movies_path = os.path.join(raw_data_path, 'movie.csv')
    
    # Vérification
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError(f" Erreur : Fichiers introuvables dans {raw_data_path}")
        
    # Chargement
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # Fusion de base pour avoir les titres avec les notes
    df_merged = pd.merge(ratings, movies, on='movieId')
    
    print(f"    Données chargées : {len(df_merged)} lignes.")
    return df_merged

def process_features(df_clean, save_path='data/processed', raw_path='data/raw'):
    """
    1. Filtre les données (utilisateurs/films actifs).
    2. Crée la matrice sparse (Utilisateur-Film).
    3. Traite les TAGS pour le Content-Based.
    4. Sauvegarde le tout (.npz, .pkl).
    """
    print("--- [Data] Traitement des features & Tags ---")

    # ==========================================
    # 1. FILTRAGE ET CRÉATION MATRICE
    # ==========================================
    print("   1. Filtrage et création matrice...")
    
    # Filtre Films (au moins 50 notes)
    movie_counts = df_clean['title'].value_counts()
    popular_movies = movie_counts[movie_counts >= 50].index
    df_filtered = df_clean[df_clean['title'].isin(popular_movies)]

    # Filtre Utilisateurs (au moins 50 notes)
    user_counts = df_filtered['userId'].value_counts()
    active_users = user_counts[user_counts >= 50].index
    df_final = df_filtered[df_filtered['userId'].isin(active_users)]

    # Catégorisation pour la matrice
    user_ids = df_final['userId'].astype('category')
    movie_titles = df_final['title'].astype('category')

    # Création de la matrice creuse
    user_item_matrix = csr_matrix((df_final['rating'], 
                                   (user_ids.cat.codes, movie_titles.cat.codes)))

    # ==========================================
    # 2. SAUVEGARDE MATRICE & MAPPINGS
    # ==========================================
    os.makedirs(save_path, exist_ok=True)
    
    # Sauvegarde Matrice, Mappings, CSV propre
    
    sparse.save_npz(os.path.join(save_path, 'user_item_matrix.npz'), user_item_matrix)
    
    mappings = {
        'user_labels': user_ids.cat.categories,
        'movie_labels': movie_titles.cat.categories
    }
    with open(os.path.join(save_path, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)
         
    df_final.to_csv(os.path.join(save_path, 'clean_data.csv'), index=False)

    # ==========================================
    # 3. TRAITEMENT DES TAGS (Pour Content-Based)
    # ==========================================
    print("   2. Traitement des Tags...")
    tag_dict = {}
    tags_file = os.path.join(raw_path, 'tag.csv')
    movies_file = os.path.join(raw_path, 'movie.csv')

    if os.path.exists(tags_file) and os.path.exists(movies_file):
        try:
            tags_df = pd.read_csv(tags_file)
            movies_df = pd.read_csv(movies_file)
            
            # Nettoyage
            tags_df['tag'] = tags_df['tag'].astype(str).str.lower()
            
            # Fusion
            merged = pd.merge(tags_df, movies_df, on='movieId')
            
            # On ne garde que les films valides (ceux qui sont dans la matrice finale)
            valid_titles = set(movie_titles.cat.categories)
            merged = merged[merged['title'].isin(valid_titles)]

            # Groupement : Top 5 tags par film
            top_tags = merged.groupby('title')['tag'].apply(lambda x: x.value_counts().index[:5].tolist())
            tag_dict = top_tags.to_dict()
            
            # Sauvegarde
            with open(os.path.join(save_path, 'movie_tags.pkl'), 'wb') as f:
                pickle.dump(tag_dict, f)
            print("    Tags traités et sauvegardés.")
            
        except Exception as e:
            print(f"    Erreur tags : {e}")
    else:
        print("    Fichiers tags manquants.")

    return user_item_matrix, mappings
import sys
import os
import random
import numpy as np

# Configuration Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# NOUVEAUX IMPORTS FACTORISÉS
from src.utils import load_artifacts
from src.evaluation import get_user_history, get_cluster_vibe, calculate_rmse, format_tags

# Chemins
PROCESSED_PATH = os.path.join(current_dir, 'data/processed')
MODELS_PATH = os.path.join(current_dir, 'models')

def run_dashboard():
    # 1. CHARGEMENT
    matrix, mappings, tag_dict, hybrid, svd, content = load_artifacts(PROCESSED_PATH, MODELS_PATH)
    
    user_labels = mappings['user_labels']
    movie_labels = mappings['movie_labels']
    user_to_idx = {u: i for i, u in enumerate(user_labels)}
    
    # 2. SÉLECTION UTILISATEUR
    test_user_id = random.choice(user_labels)
    
    
    if test_user_id not in user_to_idx:
        test_user_id = random.choice(user_labels)
        
    u_idx = user_to_idx[test_user_id]
    
    print(f"\n{'='*60}\nANALYSE DU PROFIL UTILISATEUR : {test_user_id}\n{'='*60}")

    # 3. ANALYSE PROFIL
    history = get_user_history(u_idx, matrix, movie_labels, tag_dict)
    print("SES COUPS DE COEUR :")
    for title, rating, tags in history:
        print(f" * {title} ({rating}/5)\n   Style : {tags}")

    try:
        cluster_id = hybrid.model.predict(matrix[u_idx])[0]
        vibe = get_cluster_vibe(hybrid, cluster_id, matrix, movie_labels)
        print(f"\nSON GROUPE (CLUSTER {cluster_id}) aime :\n > {', '.join(vibe)}")
    except: pass

    print(f"\n{'-'*60}\nCOMPARAISON DES MODELES\n{'-'*60}")

    # 4. MODÈLE HYBRIDE
    print("\n1. MODELE HYBRIDE")
    rmse_hybrid = calculate_rmse(hybrid, 'hybrid', matrix, user_labels, user_to_idx)
    print(f"[SCORE] RMSE : {rmse_hybrid:.4f}")
    
    recos = hybrid.recommend(test_user_id, matrix, user_to_idx, movie_labels, n_reco=5)
    for title, score in recos:
        print(f"   * {title} ({score:.2f})\n   {format_tags(title, tag_dict)}")

    # 5. MODÈLE SVD
    print("\n2. MODELE SVD")
    if svd.matrix_reconstructed is None: svd.fit(matrix)
    rmse_svd = calculate_rmse(svd, 'svd', matrix, user_labels, user_to_idx)
    print(f"[SCORE] RMSE : {rmse_svd:.4f}")
    
    recos_svd = svd.recommend(u_idx, movie_labels, n_reco=5)
    for title, score in recos_svd:
        print(f"   * {title} ({score:.2f})")

    # 6. MODÈLE CONTENT
    print("\n3. MODELE CONTENT-BASED")
    if history:
        last_liked = history[0][0]
        print(f"   Basé sur '{last_liked}'...")
        recos_content = content.recommend(last_liked, n_reco=5)
        for title in recos_content:
            print(f"   * {title}\n   {format_tags(title, tag_dict)}")
    else:
        print("   (Pas d'historique)")

if __name__ == "__main__":
    run_dashboard()
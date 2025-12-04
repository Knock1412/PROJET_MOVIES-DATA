# src/utils.py
import os
import sys
import joblib
import pickle
from scipy import sparse

def load_artifacts(processed_path, models_path):
    """Charge les matrices et les modèles."""
    print("[INFO] Chargement des artefacts...")
    try:
        matrix = sparse.load_npz(os.path.join(processed_path, 'user_item_matrix.npz'))
        
        with open(os.path.join(processed_path, 'mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
            
        tag_dict = {}
        tag_path = os.path.join(processed_path, 'movie_tags.pkl')
        if os.path.exists(tag_path):
            with open(tag_path, 'rb') as f:
                tag_dict = pickle.load(f)

        hybrid = joblib.load(os.path.join(models_path, 'kmeans_model.pkl'))
        svd = joblib.load(os.path.join(models_path, 'svd_model.pkl'))
        content = joblib.load(os.path.join(models_path, 'TF-IDF_model.pkl'))
        
        print("[INFO] Système chargé.")
        return matrix, mappings, tag_dict, hybrid, svd, content
        
    except FileNotFoundError as e:
        print(f"[ERREUR] Fichier manquant : {e}")
        sys.exit(1)
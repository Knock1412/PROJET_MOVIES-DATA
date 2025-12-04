import os
import joblib
from sklearn.decomposition import TruncatedSVD
import numpy as np

class SVDRecommender:
    """
    Système de recommandation basé sur la Factorisation de Matrice (Matrix Factorization)
    via l'algorithme TruncatedSVD.
    """
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=n_components, random_state=42)
        self.matrix_reconstructed = None

    def fit(self, user_item_matrix):
        print(f"   [SVD] Réduction de dimension à {self.n_components} composants...")
        # Transformation
        matrix_reduced = self.model.fit_transform(user_item_matrix)
        # Reconstruction (Approximation de la matrice pleine)
        self.matrix_reconstructed = self.model.inverse_transform(matrix_reduced)
        return self

    def recommend(self, user_idx, movie_labels, n_reco=5):
        if self.matrix_reconstructed is None:
            raise Exception("Modèle SVD non entraîné !")
            
        # On récupère la ligne prédite pour cet utilisateur
        preds = self.matrix_reconstructed[user_idx]
        top_idx = preds.argsort()[-n_reco:][::-1]
        
        return [(movie_labels[i], preds[i]) for i in top_idx]

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        # On ne sauvegarde pas la matrice reconstruite (trop lourd), juste le modèle
        temp = self.matrix_reconstructed
        self.matrix_reconstructed = None 
        joblib.dump(self, os.path.join(folder_path, 'svd_model.pkl'))
        self.matrix_reconstructed = temp
        print(f"   Modèle SVD sauvegardé dans {folder_path}")
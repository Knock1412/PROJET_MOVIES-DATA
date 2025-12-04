import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class KMeansRecommender:
    """
    Système de recommandation basé sur le Clustering K-Means.
    Groupe les utilisateurs similaires pour affiner les prédictions (Approche Hybride).
    """
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = None
        
    def fit(self, user_item_matrix, user_ids):
        print(f"   [KMeans] Entraînement avec {self.n_clusters} clusters...")
        self.model.fit(user_item_matrix)
        
        # On stocke le mapping User -> Cluster
        self.clusters = pd.DataFrame({
            'userId': user_ids,
            'cluster': self.model.labels_
        })
        return self

    def recommend(self, user_id, user_item_matrix, user_to_idx, movie_labels, n_reco=5):
        # 1. Trouver le cluster de l'utilisateur
        try:
            user_cluster = self.clusters[self.clusters['userId'] == user_id]['cluster'].values[0]
        except IndexError:
            return [] # Utilisateur inconnu
            
        # 2. Récupérer les voisins du même cluster
        cluster_users = self.clusters[self.clusters['cluster'] == user_cluster]['userId'].values
        cluster_indices = [user_to_idx[u] for u in cluster_users if u in user_to_idx]
        
        # 3. Filtrage Collaboratif sur ce sous-groupe
        target_idx = user_to_idx[user_id]
        sub_matrix = user_item_matrix[cluster_indices]
        
        try:
            target_rel_idx = cluster_indices.index(target_idx)
        except ValueError: return []

        # Similarité Cosinus
        sims = cosine_similarity(sub_matrix[target_rel_idx], sub_matrix).flatten()
        top_users = sims.argsort()[-51:-1] # Les 50 voisins les plus proches
        
        # Prédiction par moyenne
        preds = sub_matrix[top_users].mean(axis=0)
        preds = np.array(preds).flatten()
        
        # Masquer les films déjà vus
        seen = user_item_matrix[target_idx].toarray().flatten() > 0
        preds[seen] = 0
        
        # Trier et renvoyer
        top_idx = preds.argsort()[-n_reco:][::-1]
        return [(movie_labels[i], preds[i]) for i in top_idx if preds[i] > 0]

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(self, os.path.join(folder_path, 'kmeans_model.pkl'))
        print(f"   Modèle K-Means sauvegardé dans {folder_path}")
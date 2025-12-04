# src/evaluation.py
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

def get_user_history(user_idx, matrix, movie_labels, tag_dict, n=3):
    """Récupère les n films les mieux notés par l'utilisateur."""
    user_ratings = matrix[user_idx].toarray().flatten()
    top_indices = user_ratings.argsort()[::-1]
    
    history = []
    count = 0
    for idx in top_indices:
        if user_ratings[idx] >= 4.0:
            title = movie_labels[idx]
            tags = tag_dict.get(title, [])
            tags_str = ", ".join(tags[:3]) if tags else "Pas de tags"
            history.append((title, user_ratings[idx], tags_str))
            count += 1
            if count >= n: break
    return history

def get_cluster_vibe(model, cluster_id, matrix, movie_labels, n_top=3):
    """Trouve les films représentatifs d'un cluster."""
    cluster_indices = np.where(model.model.labels_ == cluster_id)[0]
    
    # Échantillonnage pour la rapidité
    if len(cluster_indices) > 1000:
        cluster_indices = np.random.choice(cluster_indices, 1000)
        
    sub_matrix = matrix[cluster_indices]
    movie_sums = sub_matrix.sum(axis=0).A1
    top_indices = movie_sums.argsort()[-n_top:][::-1]
    
    return [movie_labels[i] for i in top_indices]

def calculate_rmse(model, model_type, user_item_matrix, user_ids, user_to_idx, n_tests=50):
    """Calcule le RMSE sur un échantillon."""
    # Note : J'ai factorisé la logique de sélection aléatoire ici
    y_true, y_pred = [], []
    test_users = np.random.choice(user_ids, n_tests)
    
    for u_id in test_users:
        if u_id not in user_to_idx: continue
        u_idx = user_to_idx[u_id]
        ratings = user_item_matrix[u_idx].toarray().flatten()
        movies_rated = np.where(ratings > 0)[0]
        if len(movies_rated) < 2: continue
        
        sample = np.random.choice(movies_rated, min(5, len(movies_rated)))
        
        # Logique spécifique par modèle
        if model_type == 'hybrid':
            try:
                u_cluster = model.clusters[model.clusters['userId'] == u_id]['cluster'].values[0]
                cluster_users = model.clusters[model.clusters['cluster'] == u_cluster]['userId'].values
                cluster_indices = [user_to_idx[u] for u in cluster_users if u in user_to_idx]
                
                for m_idx in sample:
                    vals = user_item_matrix[cluster_indices, m_idx].toarray().flatten()
                    valid = vals[vals > 0]
                    if len(valid) > 0:
                        y_true.append(ratings[m_idx])
                        y_pred.append(valid.mean())
            except: continue
            
        elif model_type == 'svd':
            if model.matrix_reconstructed is None: continue
            for m_idx in sample:
                y_true.append(ratings[m_idx])
                y_pred.append(model.matrix_reconstructed[u_idx, m_idx])

    if not y_true: return 0.0
    return sqrt(mean_squared_error(y_true, y_pred))

def format_tags(title, tag_dict):
    tags = tag_dict.get(title, [])
    if not tags: return ""
    return f"   Tags : {', '.join(tags[:4])}"
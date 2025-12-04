import sys
import os
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.data import load_data, process_features
from src.models import KMeansRecommender, SVDRecommender, TFIDFRecommender
from src.visualization import (
    plot_elbow_curve, 
    plot_clusters_2d, 
    plot_svd_variance, 
    plot_top_tags, 
    plot_rating_distribution, 
    plot_long_tail
)

def main():
    print("\nDÉMARRAGE DU PIPELINE D'ENTRAÎNEMENT (ORGANISÉ)")
    print("="*60)
    
    # Définition des dossiers
    raw_path = os.path.join(current_dir, 'data/raw')
    processed_path = os.path.join(current_dir, 'data/processed')
    models_path = os.path.join(current_dir, 'models')
    
    # --- DÉFINITION DES DOSSIERS GRAPHIQUES ---
    figures_root = os.path.join(current_dir, 'reports/figures')
    
    # Sous-dossiers spécifiques
    kmeans_figs = os.path.join(figures_root, 'kmeans')
    svd_figs = os.path.join(figures_root, 'svd')
    content_figs = os.path.join(figures_root, 'content_based')
    
    # ---------------------------------------------------------
    # ÉTAPE 1 : ANALYSE GÉNÉRALE (Racine de reports/figures)
    # ---------------------------------------------------------
    print("\nÉTAPE 1 : Analyse Exploratoire")
    df_raw = load_data(raw_data_path=raw_path)
    
    # Ces graphiques restent à la racine car ils concernent tout le dataset
    plot_rating_distribution(df_raw, save_dir=figures_root)
    plot_long_tail(df_raw, save_dir=figures_root)
    print(f" Stats globales sauvegardées dans {figures_root}")

    # ---------------------------------------------------------
    # ÉTAPE 2 : TRANSFORMATION
    # ---------------------------------------------------------
    print("\nÉTAPE 2 : Préparation")
    user_item_matrix, mappings = process_features(df_raw, save_path=processed_path, raw_path=raw_path)
    
    # ---------------------------------------------------------
    # ÉTAPE 3 : MODÈLE HYBRIDE (Dossier /kmeans)
    # ---------------------------------------------------------
    print("\nÉTAPE 3 : Modèle K-Means")
    
    inertias = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans_test = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=10)
        kmeans_test.fit(user_item_matrix)
        inertias.append(kmeans_test.inertia_)
    
    # On sauvegarde dans le sous-dossier kmeans
    plot_elbow_curve(K_range, inertias, save_dir=kmeans_figs)
    
    hybrid_model = KMeansRecommender(n_clusters=4)
    hybrid_model.fit(user_item_matrix, mappings['user_labels'])
    hybrid_model.save(models_path)
    hybrid_model.clusters.to_csv(os.path.join(processed_path, 'user_clusters.csv'), index=False)
    
    # On sauvegarde dans le sous-dossier kmeans
    plot_clusters_2d(user_item_matrix, hybrid_model.model.labels_, save_dir=kmeans_figs)

    # ---------------------------------------------------------
    # ÉTAPE 4 : MODÈLE SVD (Dossier /svd)
    # ---------------------------------------------------------
    print("\nÉTAPE 4 : Modèle SVD")
    svd_model = SVDRecommender(n_components=20)
    svd_model.fit(user_item_matrix)
    svd_model.save(models_path)
    
    # On sauvegarde dans le sous-dossier svd
    plot_svd_variance(svd_model.model, save_dir=svd_figs)

    # ---------------------------------------------------------
    # ÉTAPE 5 : MODÈLE CONTENT (Dossier /content_based)
    # ---------------------------------------------------------
    print("\nÉTAPE 5 : Modèle Content-Based")
    try:
        movies_raw = pd.read_csv(os.path.join(raw_path, 'movie.csv'))
        tags_raw = pd.read_csv(os.path.join(raw_path, 'tag.csv'))
        
        content_model = TFIDFRecommender(max_features=5000)
        content_model.fit(movies_raw, tags_raw)
        content_model.save(models_path)
        
        # On sauvegarde dans le sous-dossier content_based
        plot_top_tags(tags_raw, save_dir=content_figs)
        
    except FileNotFoundError:
        print("  Fichiers manquants pour Content-Based.")
    
    print("\n" + "="*60)
    print(f" Graphiques enregistré dans {figures_root}")
    

if __name__ == "__main__":
    main()
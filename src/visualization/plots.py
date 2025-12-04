import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import TruncatedSVD


sns.set_theme(style="whitegrid")

def plot_elbow_curve(k_range, inertias, save_dir='reports/figures'):
    """
    Trace la courbe du coude pour déterminer le nombre optimal de clusters (K).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', markersize=8)
    plt.xlabel('Nombre de Clusters (K)')
    plt.ylabel('Inertie (Somme des distances au carré)')
    plt.title('Méthode du Coude (Elbow Method)')
    plt.grid(True)
    
    save_path = os.path.join(save_dir, 'elbow_method.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Graphique sauvegardé : {save_path}")

def plot_clusters_2d(user_item_matrix, cluster_labels, save_dir='reports/figures'):
    """
    Calcule la SVD et affiche le nuage de points des clusters.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("   [Plots] Calcul de la projection 2D (SVD)...")
    svd = TruncatedSVD(n_components=2, random_state=42)
    matrix_2d = svd.fit_transform(user_item_matrix)

    df_viz = pd.DataFrame(matrix_2d, columns=['Component 1', 'Component 2'])
    df_viz['Cluster'] = cluster_labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='Component 1', 
        y='Component 2', 
        hue='Cluster', 
        data=df_viz, 
        palette='viridis', 
        alpha=0.6,
        s=50
    )
    plt.title('Visualisation des Clusters d\'Utilisateurs')
    
    save_path = os.path.join(save_dir, 'clusters_visualization.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Graphique sauvegardé : {save_path}")

def plot_svd_variance(svd_model, save_dir='reports/figures'):
    """
    Affiche la courbe de variance expliquée cumulée pour la SVD.
    Montre combien d'information est conservée par le modèle.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calcul de la variance cumulée
    variance = np.cumsum(svd_model.explained_variance_ratio_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(variance) + 1), variance, 'r-o', linewidth=2)
    plt.xlabel('Nombre de Composants (Latent Features)')
    plt.ylabel('Variance Expliquée Cumulée')
    plt.title('Performance du modèle SVD (Information conservée)')
    plt.grid(True)
    
    save_path = os.path.join(save_dir, 'svd_variance.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Graphique SVD sauvegardé : {save_path}")

def plot_top_tags(tags_df, n=20, save_dir='reports/figures'):
    """
    Affiche un diagramme en barres des tags les plus fréquents.
    Utile pour visualiser sur quoi se base le modèle Content-Based.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Comptage des tags (en s'assurant qu'ils sont en string)
    top_tags = tags_df['tag'].astype(str).value_counts().head(n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_tags.values, y=top_tags.index, palette='magma')
    plt.title(f'Top {n} des Tags utilisés (Modèle Content-Based)')
    plt.xlabel('Nombre d\'occurrences')
    
    save_path = os.path.join(save_dir, 'content_tags_distribution.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Graphique Content sauvegardé : {save_path}")

def plot_rating_distribution(df, save_dir='reports/figures'):
    """
    Affiche la distribution des notes (Histogramme).
    Permet de voir si les utilisateurs notent sévèrement ou généreusement.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='rating', data=df, hue='rating', legend=False, palette='viridis')
    plt.title('Distribution des Notes Utilisateurs')
    plt.xlabel('Note')
    plt.ylabel('Nombre de votes')
    
    save_path = os.path.join(save_dir, 'rating_distribution.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Graphique Stats sauvegardé : {save_path}")

def plot_long_tail(df, save_dir='reports/figures'):
    """
    Affiche la courbe de popularité des films (Long Tail).
    Montre la disparité entre blockbusters et films de niche.
    """
    os.makedirs(save_dir, exist_ok=True)

    movie_counts = df.groupby('title')['rating'].count().sort_values(ascending=False).values
    
    plt.figure(figsize=(10, 6))
    plt.plot(movie_counts, color='blue')
    plt.title('Popularité des films (La Longue Traîne)')
    plt.xlabel('Films (du plus populaire au moins populaire)')
    plt.ylabel('Nombre de notes')
    plt.fill_between(range(len(movie_counts)), movie_counts, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'long_tail_popularity.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   Graphique Stats sauvegardé : {save_path}")
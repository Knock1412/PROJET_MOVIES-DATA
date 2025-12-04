import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class TFIDFRecommender:
    """
    Système de recommandation basé sur le contenu (Content-Based)
    utilisant la vectorisation TF-IDF sur les Genres et Tags.
    """
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        self.tfidf_matrix = None
        self.movies_df = None

    def fit(self, movies_df, tags_df):
        print("   [TF-IDF] Vectorisation des métadonnées (Tags + Genres)...")
        # Préparation du texte
        tags_df['tag'] = tags_df['tag'].astype(str).str.lower()
        tags_grouped = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        
        self.movies_df = pd.merge(movies_df, tags_grouped, on='movieId', how='left').fillna('')
        # On crée une "soupe" de mots-clés : Genres + Tags
        self.movies_df['metadata'] = self.movies_df['genres'].str.replace('|', ' ') + ' ' + self.movies_df['tag']
        
        # Calcul du TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['metadata'])
        return self

    def recommend(self, movie_title, n_reco=5):
        # Trouver l'index du film
        indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()
        
        if movie_title not in indices:
            return []
            
        idx = indices[movie_title]
        
        # Calcul de similarité cosinus (linear_kernel est plus rapide ici)
        sim_scores = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # Trier
        top_idx = sim_scores.argsort()[-(n_reco+1):-1][::-1]
        
        return self.movies_df['title'].iloc[top_idx].tolist()

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(self, os.path.join(folder_path, 'TF-IDF_model.pkl'))
        print(f"    Modèle TF-IDF sauvegardé dans {folder_path}")
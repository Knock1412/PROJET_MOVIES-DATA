# Système de Recommandation de Films Hybride (MovieLens)

## Description du Projet

Ce projet a été réalisé dans le cadre du Master 1 Informatique Big Data (Université Paris 8). Il vise à concevoir et comparer plusieurs approches de systèmes de recommandation pour prédire les préférences cinématographiques des utilisateurs.

L'architecture principale repose sur un **Modèle Hybride** innovant combinant apprentissage non supervisé (Clustering) et filtrage collaboratif. Ce modèle est mis en compétition avec deux standards de l'industrie : la Factorisation de Matrice (SVD) et l'approche basée sur le contenu (Content-Based).

## Architecture du Projet

Le code respecte les standards de développement "Data Science" avec une factorisation complète des modules dans le dossier `src`.

PROJET_MOVIES-DATA/
├── data/
│   ├── raw/               # Dossier pour les fichiers CSV sources (non inclus dans le git)
│   └── processed/         # Dossier pour les matrices creuses (.npz) et mappings (.pkl)
├── models/                # Stockage des modèles entraînés (.pkl)
├── notebooks/             # Notebooks Jupyter pour la démonstration et l'analyse
├── reports/
│   └── figures/           # Graphiques générés automatiquement (PNG)
├── src/                   # Code source (Package Python)
│   ├── data/              # Scripts de chargement et transformation (ETL)
│   ├── models/            # Classes des algorithmes (KMeans, SVD, TF-IDF)
│   ├── visualization/     # Scripts de génération des graphiques
│   ├── evaluation.py      # Fonctions de calcul de métriques (RMSE)
│   └── utils.py           # Fonctions utilitaires de chargement
├── train.py               # Script principal d'entraînement et de sauvegarde
├── predict.py             # Script de tableau de bord de prédiction
├── README.md              # Documentation technique
└── requirements.txt       # Liste des dépendances logicielles

## Installation et Configuration

### 1. Prérequis
Ce projet nécessite Python 3.8 ou supérieur.

### 2. Installation des dépendances
Exécutez la commande suivante à la racine du projet :

pip install -r requirements.txt

### 3. Téléchargement des Données (Important)
En raison de la taille du jeu de données MovieLens (plusieurs centaines de Mo), les fichiers sources ne sont pas hébergés sur ce dépôt GitHub.

1. Téléchargez le dataset **"MovieLens 20M Dataset"** :
   [https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset]

2. Décompressez l'archive.

3. Placez les fichiers suivants dans le dossier `data/raw/` du projet :
   - `rating.csv`
   - `movie.csv`
   - `tag.csv`

L'arborescence doit être : `PROJET_MOVIES-DATA/data/raw/rating.csv`.

## Guide d'Utilisation

Le projet est entièrement automatisé via deux scripts principaux.

### Phase 1 : Entraînement (train.py)
Ce script lance le pipeline complet : chargement, nettoyage, analyse exploratoire, entraînement des trois modèles et génération des rapports graphiques.

Commande :
python train.py

Résultats attendus :
- Création des fichiers dans data/processed/
- Sauvegarde des modèles dans models/
- Génération des graphiques d'analyse dans reports/figures/

### Phase 2 : Prédiction et Comparaison (predict.py)
Ce script lance un tableau de bord interactif dans le terminal. Il sélectionne un utilisateur aléatoire, analyse son profil, et compare les recommandations des trois modèles en temps réel.

Commande :
python predict.py

Fonctionnalités du tableau de bord :
- Affichage de l'historique et du cluster de l'utilisateur.
- Calcul du RMSE (Root Mean Squared Error) pour évaluer la précision.
- Affichage du "Podium" final désignant le meilleur modèle pour cet utilisateur.

## Méthodologie Scientifique

Le projet implémente et compare trois stratégies :

### 1. Modèle Hybride (K-Means + Filtrage Collaboratif)
* **Approche :** Segmentation des utilisateurs en clusters homogènes (K-Means) avant d'appliquer un filtrage collaboratif (User-Based) restreint aux membres du cluster.
* **Objectif :** Réduire le bruit (noise reduction) et améliorer la pertinence locale des suggestions.
* **Fichier source :** src/models/kmeans.py

### 2. Modèle SVD (Singular Value Decomposition)
* **Approche :** Réduction de dimensionnalité mathématique (Matrix Factorization) pour identifier les facteurs latents reliant utilisateurs et films.
* **Objectif :** Minimiser l'erreur mathématique globale (RMSE). Sert de modèle de référence (baseline).
* **Fichier source :** src/models/truncated_svd.py

### 3. Modèle Content-Based (TF-IDF)
* **Approche :** Analyse sémantique des métadonnées (Tags et Genres) via vectorisation TF-IDF.
* **Objectif :** Recommander des films similaires textuellement. Résout le problème du "Cold Start" (nouveaux utilisateurs sans historique).
* **Fichier source :** src/models/tfidf.py

## Métriques et Évaluation

La performance est mesurée via le **RMSE** (Root Mean Squared Error).
- Un RMSE plus bas indique une meilleure précision de prédiction.
- Les résultats montrent généralement que l'approche Hybride offre un excellent compromis entre la précision mathématique du SVD et l'explicabilité des clusters.

## Auteur

**Adrien LIMACHE**
M1 Informatique Big Data - Université Paris 8

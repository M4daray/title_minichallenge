from sentence_transformers import SentenceTransformer
import pandas as pd
from title import PATH_PRIMARY, PATH_REPORT
import os
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score


def find_best_nb_clust():
    df = pd.read_parquet(os.path.join(PATH_PRIMARY, 'juritrack_emb.parquet'))
    # df = df[0:1000]
    X = np.array([x.tolist() for x in df['embeddings_ordalie'].tolist()])

    scores = []
    for n in range(8, 100):
        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto")
        clusteurs = kmeans.fit_predict(X)
        score = silhouette_score(X, kmeans.fit_predict(X))
        scores.append({'nb_clusteur': n, 'silhouette_score': score})
        print(f'nb_clusteur: {n}, silhouette_score: {score=}')

    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(os.path.join(PATH_REPORT, 'kmean.csv'), index=False)

def assign_clust():
    """ """
    n: int = 98
    df = pd.read_parquet(os.path.join(PATH_PRIMARY, 'juritrack_emb.parquet'))
    X = np.array([x.tolist() for x in df['embeddings_ordalie'].tolist()])
    kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto")
    clusteurs = list(kmeans.fit_predict(X))
    print(clusteurs)
    df['target'] = clusteurs
    df.to_parquet(os.path.join(PATH_PRIMARY, 'juritrack_target.parquet'))

if __name__ == "__main__":
    assign_clust()
from sentence_transformers import SentenceTransformer
import pandas as pd
from title import PATH_PRIMARY, PATH_REPORT
import os
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from natsort import natsort_keygen

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
    titles = df['title'].tolist()

    # Fit cluster
    X = np.array([x.tolist() for x in df['embeddings_ordalie'].tolist()])
    kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto")
    clusteurs = list(kmeans.fit_predict(X))
    df['target'] = [str(int(x)) for x in clusteurs]

    # Best candidate to represent the cluster
    res = []
    for centroid in df['target'].unique():
        distance_matrix = kmeans.transform(X)[:, int(centroid)]
        nearest_sample_from_centroid: int = np.argsort(distance_matrix)[0]
        res.append({'idx_clusteur': centroid, 'nearest_sample': titles[nearest_sample_from_centroid]})

    # Write results
    df_clust = pd.DataFrame(res).sort_values(by=['idx_clusteur'], key=natsort_keygen())
    df_clust.to_csv(os.path.join(PATH_PRIMARY, 'clusteurs_info.csv'), index=False)
    df.to_parquet(os.path.join(PATH_PRIMARY, 'juritrack_target.parquet'))

if __name__ == "__main__":
    # find_best_nb_clust()
    assign_clust()
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List
from title import PATH_PRIMARY, PATH_REPORT, PATH_RAW
import os
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(PATH_PRIMARY, 'juritrack_target.parquet'))
    # df = df[0:1000]

    y = df['target'].tolist()
    X = df['title'].tolist()

    with open(os.path.join(PATH_RAW, 'stopwords.txt')) as f:
        stop_words: List[str] = list(f.readlines())

    vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, max_df=0.8, min_df=5, ngram_range=(1,3), stop_words=stop_words)
    svm = LinearSVC(dual="auto", random_state=0, tol=1e-5)
    components = [("tfidf", vectorizer),
        ("svm", svm)]
    clf = Pipeline(components)

    y_pred = cross_val_predict(clf, X, y, cv=5)
    df['pred'] = y_pred

    report = classification_report(df['target'], df['pred'], output_dict=True)
    df_report = pd.DataFrame(report).T.reset_index()
    df_report.to_csv(os.path.join(PATH_REPORT, 'classification_report.csv'), index=False)


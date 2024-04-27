from sentence_transformers import SentenceTransformer
import pandas as pd
from title import PATH_PRIMARY
import os
import pandas as pd
from title import PATH_RAW, PATH_INTERMEDIATE
from sklearn.manifold import TSNE


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(PATH_PRIMARY, 'juritrack_emb.parquet'))
    X = df['embeddings_ordalie'].tolist()
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
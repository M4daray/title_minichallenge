from sentence_transformers import SentenceTransformer
import pandas as pd
from title import PATH_INTERMEDIATE, PATH_PRIMARY
import os



if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(PATH_INTERMEDIATE, 'juritrack.parquet'))
    model_name = "OrdalieTech/Solon-embeddings-large-0.1"
    # df = df[0:100]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['title'].tolist(), show_progress_bar=True, batch_size=128, convert_to_numpy=True)
    # print(embeddings.tolist())
    df['embeddings_ordalie'] = embeddings.tolist()
    df.to_parquet(os.path.join(PATH_PRIMARY, 'juritrack_emb.parquet'))
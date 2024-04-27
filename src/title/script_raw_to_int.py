import os
import pandas as pd
from title import PATH_RAW, PATH_INTERMEDIATE

df = pd.read_csv(os.path.join(PATH_RAW, 'juritrack_arretes_titles_01012024_22012024.csv'))
df.columns = [x.strip().lower().replace(' ', '_') for x in df.columns]
df = df[['title']]
df = df.reset_index().rename(columns={'index': 'id_title'})
print(df.columns)
df.to_parquet(os.path.join(PATH_INTERMEDIATE, 'juritrack.parquet'))
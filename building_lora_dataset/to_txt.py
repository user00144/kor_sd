import pandas as pd
import os
from tqdm import tqdm


DATA_PATH = "/workspace/data/korean_images/dataset/crop"
df_trans = pd.read_csv('metadata_trans.csv')


for i in tqdm(range(len(df_trans))) :
    fn = df_trans.iloc[i].file_name
    origin_fn = fn[:-4]
    txt_fn = origin_fn + ".txt"
    caption = f"{df_trans.iloc[i].paper_type}, {df_trans.iloc[i].method}, {df_trans.iloc[i].caption_trans}"
    with open(os.path.join(DATA_PATH, txt_fn), "w") as txt_file :
        txt_file.write(caption)
import pandas as pd
import os
from tqdm import tqdm


DATA_PATH = "/workspace/data/korean_images/dataset/crop"
df_trans = pd.read_csv('metadata_trans.csv')

fns = []
captions = []
for i in tqdm(range(len(df_trans))) :
    fn = df_trans.iloc[i].file_name
    caption = f"{df_trans.iloc[i].paper_type}, {df_trans.iloc[i].method}, {df_trans.iloc[i].caption_trans}"
    captions.append(caption)
    fns.append(fn)
    

df_out = pd.DataFrame()

df_out['file_name'] = fns
df_out['ko_cap'] = captions

df_out.to_csv(os.path.join(DATA_PATH, "metadata.csv"), index = False)
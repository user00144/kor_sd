import deepl
import pandas as pd
from tqdm import tqdm

auth_key = ""
translator = deepl.Translator(auth_key)

df_metadata = pd.read_csv('metadata_cap.csv')


translated = []

for i in tqdm(range(len(df_metadata))) :
    caption = df_metadata.iloc[i].caption
    result = translator.translate_text(caption, target_lang = "ko")
    translated.append(result)
    
df_metadata['caption_trans'] = translated
df_metadata.to_csv('metadata_trans.csv', index = False)
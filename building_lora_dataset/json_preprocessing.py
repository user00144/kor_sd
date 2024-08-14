import os
from tqdm import tqdm
import json
import pandas as pd

ROOT_PATH = "/workspace/data/korean_images/metadata"
DATA_PATH = os.path.join(ROOT_PATH, "TL_ALL")

list_file = os.listdir(DATA_PATH)

origin_file_names = []
paper_type = []
method = []

def get_paper_type(m) :
    if m == 1 :
        return "화선지"
    else :
        return "순지"

def get_method(m) :
    if m <= 3 and m > 0 :
        return "백묘법"
    elif m <= 7 and m > 3 :
        return "구륵법"
    else :
        return "몰골법" 
    
    
for file_name in tqdm(list_file) :
    with open(os.path.join(DATA_PATH, file_name)) as f :
        data = json.load(f)
        for i in range(2) :
            fname = data['annotation']['Paire']
            fname = f"{i}_" + fname
            origin_file_names.append(fname)
            paper_type.append(get_paper_type(data['annotation']['PaperType']))
            method.append(get_method(data['annotation']['Paint']['Method']))


df_out = pd.DataFrame()

df_out['file_name'] = origin_file_names
df_out['paper_type'] = paper_type
df_out['method'] = method

print(df_out)
df_out.to_csv('metadata.csv', index = False)
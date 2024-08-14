from qdrant_client.http.models import PointStruct
import pandas as pd
from tqdm import tqdm
from configure import DATA_PATH, COLLECTION_NAME

class Upserter() :
    def __init__(self, TE, db_client) :
        print("Loading Datasets ...")
        self.df_prompts = pd.read_parquet(DATA_PATH)    
        
        #sample 300,000   
        self.df_prompts = self.df_prompts.sample(300000)
        
        self.TE = TE
        self.db_client = db_client
        print("! Complete !\n")
        #df_prompts = pd.read_csv(os.path.join(DATA_PATH), nrows = 100000)

    def upsert(self) :
        prompts_data = self.df_prompts['prompt'].values
        length_prompts = len(prompts_data)
        print('prompts dataset length : ',length_prompts)
        print("Upserting Datasets to DB ...")
        points = []
        for i in tqdm(range(length_prompts)) :
            str_prompt = prompts_data[i]
            embed_prompt = self.TE.embed_text(str_prompt)
            point = PointStruct(
                id = i,
                vector = embed_prompt.tolist(),
                payload = {"prompt" : str_prompt}
            )
            points.append(point)
            
        operation_info = self.db_client.upsert(
            collection_name = COLLECTION_NAME,
            wait = True,
            points = points
        )
        print("! Complete !\n")
        return operation_info

    
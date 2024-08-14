import time
from tqdm import tqdm
import pandas as pd
from configure import EXP_DATA_PATH, COLLECTION_NAME

class Query_system() :
    def __init__(self, TE, db_client) :
        self.TE = TE
        self.db_client = db_client
        self.df_prompt_dataset = pd.read_csv(EXP_DATA_PATH)


    def search_query(self, query) :
        start = time.time()
        vector = self.TE.embed_text(query)
        results = self.db_client.search(
            collection_name = COLLECTION_NAME, query_vector = vector, limit = 3
        )

        end = time.time()
        out_prompts = []
        
        for result in results :
            out_prompts.append(result.payload["prompt"])
        
        search_time = end - start
        return out_prompts, search_time
        
        
    def exp_(self):
        print('Experiment start ...')
        queries = []
        out_prompts_1 = []
        out_prompts_2 = []
        out_prompts_3 = []
        search_times = []

        for i in tqdm(range(len(self.df_prompt_dataset))) :
            query = self.df_prompt_dataset.iloc[i]['default']
            out_prompt , search_time = self.search_query(query)
            queries.append(query)
            out_prompts_1.append(out_prompt[0])
            out_prompts_2.append(out_prompt[1])
            out_prompts_3.append(out_prompt[2])

            search_times.append(search_time)
            
            
        out_df = pd.DataFrame()
        out_df['search_query'] = queries
        out_df['out_1'] = out_prompts_1
        out_df['out_2'] = out_prompts_2
        out_df['out_3'] = out_prompts_3
        out_df['search_times'] = search_times

        out_df.to_csv('exp_vectordb.csv',index = False)
        print('! Complete !\n')

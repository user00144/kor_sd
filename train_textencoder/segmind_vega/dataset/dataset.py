from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import json
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class Text_Dataset(Dataset) :
    def __init__(self, data, diff, CONFIG):
        super().__init__()
        self.data = data
        self.ko_tokenizer = AutoTokenizer.from_pretrained(CONFIG.TOKENIZER_MODEL_NAME)
        self.en_tokenizer = diff.tokenizer

        
    def __getitem__(self, index):
        row = self.data.iloc[index]
        ko_data = row['ko']
        en_data = row['en']
        
        ko_token, en_ko_token, en_en_token = self.tokenize(ko_data, en_data)
        
        return ko_token, en_ko_token, en_en_token
    
    def tokenize(self, ko_data, en_data) :
        ko_token = self.ko_tokenizer(ko_data, padding = 'max_length' , return_tensors="pt", truncation=True)
        en_ko_token = self.ko_tokenizer(en_data, padding = 'max_length', return_tensors="pt" , truncation=True)
        en_en_token = self.en_tokenizer(en_data, padding = 'max_length', return_tensors="pt" , truncation=True)
        
        ko_token['input_ids'] = ko_token['input_ids'].squeeze()
        ko_token['attention_mask'] = ko_token['attention_mask'].squeeze()
        
        en_ko_token['input_ids'] = en_ko_token['input_ids'].squeeze()
        en_ko_token['attention_mask'] = en_ko_token['attention_mask'].squeeze()
        
        en_en_token['input_ids'] = en_en_token['input_ids'].squeeze()
        en_en_token['attention_mask'] = en_en_token['attention_mask'].squeeze()
                
        return ko_token, en_ko_token, en_en_token
    
    
    def __len__(self) :
        return len(self.data)
    
    
def load_json_data(datapath) :
    with open(datapath) as f :
        data_dict = json.load(f)
        
    return data_dict

def load_csv_data(datapath) :
    df = pd.read_csv(datapath)
    return df


def get_dataloader(CONFIG, diff) :
    data = load_csv_data(CONFIG.DATA_PATH)
    #data = load_json_data(CONFIG.DATA_PATH)
    data = data.sample(1500000)
    tr_data, val_data = train_test_split(data, random_state=CONFIG.SEED, test_size = CONFIG.VAL_SIZE)
    train_dataset = Text_Dataset(tr_data, diff, CONFIG)
    val_dataset = Text_Dataset(val_data, diff, CONFIG)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader
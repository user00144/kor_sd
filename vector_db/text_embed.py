from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from configure import MODEL_NAME
import torch

class Text_Embeder () :
    def __init__(self) :
        print(f"Loading Embedding Models ... MODEL_NAME : {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            device_map="auto",
        )
        print("! Complete !\n")
        
    
    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        inputs = inputs.to("cuda")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(1).detach().cpu().numpy()[0]
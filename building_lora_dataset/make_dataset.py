import pandas as pd
from datasets import Dataset, Features, Value, Image, DatasetDict
from PIL import Image as PILImage
from huggingface_hub import login


TOKEN = ""
login(TOKEN)

df = pd.read_csv('metadata.csv')
df = df.rename(columns={'file_name': 'image'})
dataset = Dataset.from_pandas(df)

def load_image(e) :
    img_path = e['image']
    e['image'] = PILImage.open(img_path)
    return e

dataset = dataset.map(load_image)

features = Features({
    'image' : Image(),
    'ko_cap' : Value('string')
})

dataset = dataset.cast(features)

dataset_dict = DatasetDict({"train" : dataset})

print(dataset_dict)

dataset.save_to_disk("/workspace/data/korean_images/dataset/lora_dataset")
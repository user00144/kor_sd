from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
model.to("cuda")

DATA_PATH = "/workspace/data/korean_images/dataset/crop"
df_metadata = pd.read_csv('metadata.csv')

def get_caption(img_path) :
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to("cuda")
    outputs = model.generate(**inputs, do_sample = False ,num_beams=3 , max_length=75, min_length=5)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

captions = []
drop_idx = []
for i in tqdm(range(len(df_metadata))) :
    try :
        caption = get_caption(os.path.join(DATA_PATH, df_metadata.iloc[i].file_name))
        captions.append(caption)
    except :
        drop_idx.append(i)
        continue

df_metadata.drop(drop_idx ,inplace=True)
df_metadata.reset_index(drop = True, inplace=True)
df_metadata['caption'] = captions
df_metadata.to_csv('metadata_cap.csv', index = False)
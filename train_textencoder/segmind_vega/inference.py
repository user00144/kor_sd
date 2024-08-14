from diffusers import StableDiffusionXLPipeline
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_KOCLIP_PATH = "/workspace/data/models/koclip_textmodel_1"
PROJECTION_KOCLIP_PATH = "/workspace/dev/ai_SD_finetuning/archive/pretrained_projection_seed42"

prompt = "우주선 고양이"


tokenizer = AutoTokenizer.from_pretrained(DEFAULT_KOCLIP_PATH)

defalut_text_model = AutoModel.from_pretrained(DEFAULT_KOCLIP_PATH) 
projection_text_model = AutoModel.from_pretrained(PROJECTION_KOCLIP_PATH)

diff = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

diff.text_encoder.text_model = defalut_text_model.text_model
diff.text_encoder_2.text_model = projection_text_model.text_model
diff.tokenizer = tokenizer
diff.tokenizer_2 = tokenizer

#diff.to("cuda")

image = diff(prompt = prompt,
             num_inference_steps = 10 ,
             ).images[0]

image.save("./test.png")
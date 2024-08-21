from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
import torch

print("Loading CLIP ...")
ko_text_model = CLIPTextModel.from_pretrained("/workspace/data/models/converted", torch_dtype=torch.float16)
ko_tokenizer = CLIPTokenizer.from_pretrained("/workspace/data/models/converted")
print("Loading CLIP Complete !")

print("Loading SD Pipeline ...")
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                text_encoder = ko_text_model,
                                                tokenizer = ko_tokenizer,
                                                torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
print("Loading SD Pipeline Complete !")

print("Loading LoRA Weights ...")
model.load_lora_weights("/workspace/data/Lora_weights/koreandress_female", adapter_name = "kdress")
model.load_lora_weights('./Asian Cute Face.safetensors', adapter_name = "face")
model.set_adapters(["kdress", "face"], adapter_weights=[1, 1])
#model.fuse_lora(adapter_names=["kdress", "face"], lora_scale=1.0)

print("Loading LoRA Weights Complete !")

model.to("cuda")

def get_img(prompt, neg_prompt) :
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    image = model(prompt = prompt, negative_prompt = neg_prompt).images[0]
    
    ender.record()
    torch.cuda.synchronize()
    infer_time = starter.elapsed_time(ender) / 1000

    return infer_time, image

i = 0

while (True) :
    prompt = input("prompt > ")
    neg = input("negative > ")
    infer_time , image = get_img(prompt, neg)
    print('inference time : ', infer_time, 'sec')
    image.save(f'./imgs/test_{i}.png')
    i += 1
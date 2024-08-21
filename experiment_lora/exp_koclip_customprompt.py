import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import pandas as pd

SEED = 42 #1147446245
generator = torch.Generator(device="cuda").manual_seed(SEED)

print("============================== Loading Models & Pipelines ==============================")

print("Loading CLIP Model")
clip_model = CLIPTextModel.from_pretrained("/workspace/dev/archive/converted", torch_dtype=torch.float16)
print(clip_model)
clip_tokenizer = CLIPTokenizer.from_pretrained("/workspace/dev/archive/converted")
print("Loading CLIP Model Complete!")


print("Loading SD Pipeline ...")
diff_pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                              text_encoder = clip_model,
                                              tokenizer = clip_tokenizer,
                                              torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

print("Loading SD Pipeline Complete!")

LORA_PATH = "/workspace/data/Lora_weights/koreandress_3"

print("Loading LoRA Weights ...")
diff_pipe.load_lora_weights(LORA_PATH)
print("Loading LoRA Weights Complete!")

diff_pipe.to("cuda")
refiner.to("cuda")


refine_prompt = "Realistic, Masterpiece , Photo, Super resolution , High Quality"
refine_neg = "(open mouth),[lowres],[smartphone camera], [amateur camera],[3d render],[sepia],((cartoon)),((anime)),((drawn)),(paint),(teeth), deformed, bad body proportions, mutation, (ugly), disfigured,(string)"

def get_inference(prompt, negative_prompt) :
    starter, ender1, ender2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    
    #======================== inference ========================
    
    image = diff_pipe(prompt = prompt, negative_prompt = negative_prompt, generator = generator).images[0]
    
    ender1.record()

    refined = refiner(prompt = refine_prompt, negative_prompt = refine_neg , image = image).images[0]
    
    #======================== inference end ========================

    ender2.record()
    torch.cuda.synchronize()
    pipeline_time = starter.elapsed_time(ender1) / 1000
    refine_time = starter.elapsed_time(ender2) / 1000

    return pipeline_time, refine_time, image ,refined

neg = "불완전한 신체 구조, 불명확, 잘림, 왜곡, 복사, 실수, 추가 팔, 추가 다리, 불쾌한 비율, 긴 목, 저급, 저해상도, 팔 다리 부족, 건강하지 않음, 유전적 변이, 프레임을 벗어남, 텍스트 삽입, 매력적이지 않음, 최저 품질, 불완전한 얼굴"

i = 0

while(True) :
    prompt = input('prompt >')
    pipeline_time, refine_time, image ,refined = get_inference(prompt, neg)
    path_p = f'./imgs/test_{i}.png'
    path_r = f'./refined/test_{i}.png'
    print('pipe_time : ', pipeline_time, ' refine_time : ', refine_time)
    image.save(path_p)
    refined.save(path_r)
    i += 1

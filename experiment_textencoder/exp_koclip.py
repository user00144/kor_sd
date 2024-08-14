import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline
import torch
import pandas as pd

SEED = 42 #1147446245
generator = torch.Generator(device="cuda").manual_seed(SEED)

print("============================== Loading Models & Pipelines ==============================")

# print("Loading CLIP Model")
# clip_model = CLIPTextModel.from_pretrained("/workspace/data/models/koclip_textmodel_1", torch_dtype=torch.float16)
# print(clip_model)
# clip_tokenizer = CLIPTokenizer.from_pretrained("/workspace/data/models/koclip_textmodel_1")
# print("Loading CLIP Model Complete!")


print("Loading SD Pipeline ...")
diff_pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                            #   text_encoder = clip_model,
                                            #   tokenizer = clip_tokenizer,
                                              torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
print("Loading SD Pipeline Complete!")

diff_pipe.to("cuda")



def get_inference(prompt) :
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    
    #======================== inference ========================
    
    image = diff_pipe(prompt = prompt, generator = generator).images[0]
    
    #======================== inference end ========================

    ender.record()
    torch.cuda.synchronize()
    infer_time = starter.elapsed_time(ender) / 1000

    return infer_time, image

def save_imgs(prompt, enhanced_prompt, ids) :
    prompt_infer_time, prompt_image = get_inference(prompt)
    enhanced_infer_time, enhanced_image = get_inference(enhanced_prompt)
    
    prompt_out_dir = f"./outs/default/default_{ids}_seed{SEED}.png"
    enhanced_out_dir = f"./outs/enhanced/enhanced_{ids}_seed{SEED}.png"
    prompt_image.save(prompt_out_dir)
    enhanced_image.save(enhanced_out_dir)
    
    return prompt_infer_time, prompt_out_dir, enhanced_infer_time, enhanced_out_dir

def get_csv_outs(df) :
    p_inference_time = []
    p_out_dir = []
    e_inference_time = []
    e_out_dir = []
    
    for i in range(len(df)) :
        ids = df.iloc[i]['ID']
        print(f"=== {ids} ===")
        prompt_infer_time, prompt_out_dir, enhanced_infer_time, enhanced_out_dir = save_imgs(df.iloc[i]['default'], df.iloc[i]['enhanced'], ids)
        p_inference_time.append(prompt_infer_time)
        p_out_dir.append(prompt_out_dir)
        e_inference_time.append(enhanced_infer_time)
        e_out_dir.append(enhanced_out_dir)
        print('p_it : ', prompt_infer_time, ' e_it : ', enhanced_infer_time)
    
    df_outs = df.copy()
    df_outs['p_inf_time'] = p_inference_time
    df_outs['p_out_dir'] = p_out_dir
    df_outs['e_inf_time'] = e_inference_time
    df_outs['e_out_dir'] = e_out_dir
    
    print(df_outs.head())
    
    return df_outs


df_prompt = pd.read_csv("./prompts_translation.csv")
df_outs = get_csv_outs(df_prompt)
df_outs.to_csv('./outs.csv', index = False)
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import DiffusionPipeline
import torch
import pandas as pd

SEED = 1147446245
generator = torch.Generator(device="cuda").manual_seed(SEED)

print("============================== Loading Models & Pipelines ==============================")

MODEL = 'gyupro/Koalpaca-Translation-KR2EN'

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
)

print("Loading Translation Pipeline ...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=AutoTokenizer.from_pretrained(MODEL),
    device='cuda'
)


def get_inference(prompt) :
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    
    #======================== inference ========================
    
    ans = pipe(
        f"### source: {prompt}\n\n### target:",
        do_sample=True,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
        )

    msg = ans[0]["generated_text"]

    if "###" in msg:
        msg = msg.split("###")[0]
    
    #======================== inference end ========================

    ender.record()
    torch.cuda.synchronize()
    infer_time = starter.elapsed_time(ender) / 1000

    return infer_time, msg

def save_imgs(prompt, enhanced_prompt) :
    prompt_infer_time, prompt_msg = get_inference(prompt)
    enhanced_infer_time, enhanced_msg = get_inference(enhanced_prompt)

    return prompt_msg, enhanced_msg
from tqdm import tqdm

def get_csv_outs(df) :
    prompt_msgs = []
    enhanced_msgs = []
    
    for i in tqdm(range(len(df))) :
        prompt_msg, enhanced_msg = save_imgs(df.iloc[i]['default'], df.iloc[i]['enhanced'])
        prompt_msgs.append(prompt_msg)
        enhanced_msgs.append(enhanced_msg)
    df_outs = df.copy()
    df_outs['default'] = prompt_msgs
    df_outs['enhanced'] = enhanced_msgs
    print(df_outs.head())
    
    return df_outs


df_prompt = pd.read_csv("./prompts.csv")
df_outs = get_csv_outs(df_prompt)
df_outs.to_csv('./prompts_translation.csv', index = False)
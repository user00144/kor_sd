from diffusers import StableDiffusionXLPipeline
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_KOCLIP_PATH = ""
PROJECTION_KOCLIP_PATH = ""

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_KOCLIP_PATH)

defalut_text_model = AutoModel.from_pretrained(DEFAULT_KOCLIP_PATH) 
projection_text_model = AutoModel.from_pretrained(PROJECTION_KOCLIP_PATH)

clip_models = [defalut_text_model.text_model, projection_text_model.text_model]
clip_tokenizers = [tokenizer, tokenizer]

def embed_fn(prompt, prompt2) :
    prompt2 = prompt2 or prompt
    
    prompts = [prompt, prompt2]
    
    prompt_embeds_list = []
    
    for prompt, tokenizer, text_encoder in zip(prompts, clip_tokenizers, clip_models):
    
        text_inputs = tokenizer(prompt,
                                    padding = 'max_length',
                                    max_length = tokenizer.model_max_length,
                                    truncation = True,
                                    return_tensors = "pt",
                                    )
        
        text_input_ids = text_inputs.input_ids
        
        with torch.no_grad() :
            prompt_embeds = text_encoder(
                text_input_ids,
                output_hidden_states = True,
            )
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            
            prompt_embeds_list.append(prompt_embeds)
            
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        
        return prompt_embeds, pooled_prompt_embeds

def prompt_to_embed(prompt, prompt2, negative_prompt, negative_prompt2) :
    
    prompt_embeds, pooled_prompt_embeds = embed_fn(prompt, prompt2)

    if negative_prompt is None :
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    else :
        negative_prompt_embeds, negative_pooled_prompt_embeds = embed_fn(negative_prompt, negative_prompt2)
        
    return pooled_prompt_embeds, prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_embeds


prompt = "우주선 고양이"

pooled_prompt_embeds, prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_embeds = prompt_to_embed(prompt, None, None , None)


diff = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,)

call_args = dict(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
)

image = diff(**call_args).images[0]

image.save("./test.png")
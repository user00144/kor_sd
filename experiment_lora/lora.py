from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import torch


SEED = 1880424069
file_name = "realDream_15SD15.safetensors"
CLIP_PATH_1 = "/workspace/data/models/koclip_textmodel_1"

# clip_model_1 = CLIPTextModel.from_pretrained(CLIP_PATH_1 , torch_dtype=torch.float16, use_safetensors=True)
# clip_tokenizer_1 = CLIPTokenizer.from_pretrained(CLIP_PATH_1)

# diff = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
#                                                text_encoder = clip_model_1,
#                                                tokenizer = clip_tokenizer_1
#                                                ,torch_dtype=torch.float16, use_safetensors=True, variant="fp16")


diff_lora = StableDiffusionPipeline.from_single_file(file_name,
                                                     #text_encoder = clip_model_1,
                                                     #tokenizer = clip_tokenizer_1,
                                                     torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

diff_lora.to("cuda")
#prompt = "붉은 머리 여자"
#negative_prompt = "(누드),(입을 벌린),[저해상도],[스마트폰 카메라],[아마추어 카메라],[3D 렌더링],[세피아],((만화)),((애니메이션)),((그린)),(페인트),(치아), 변형, 나쁜 신체 비율, 돌연변이, (추한), 변형,(끈), (문자열)."


def get_img(prompt, negative_prompt, i) :
    image = diff_lora(prompt = prompt, negative_prompt = negative_prompt).images[0]
    image.save(f"./test{i}.png")

i = 0
while(True) :
    prompt = input("pos >> ")
    negative_prompt = input("nag >> ")
    get_img(prompt, negative_prompt, i)
    i += 1
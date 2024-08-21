import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from flask import Flask, render_template, request, send_file
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer

app = Flask(__name__)

# Stable Diffusion 모델 로드
print("Loading CLIP Model")
clip_model = CLIPTextModel.from_pretrained("/workspace/dev/archive/converted", torch_dtype=torch.float16)
print(clip_model)
clip_tokenizer = CLIPTokenizer.from_pretrained("/workspace/dev/archive/converted")
print("Loading CLIP Model Complete!")


print("Loading SD Pipeline ...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                              text_encoder = clip_model,
                                              tokenizer = clip_tokenizer,
                                              torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
print("Loading SD Pipeline Complete!")

pipe.to("cuda")
refiner.to("cuda")

refine_prompt = "Realistic, Masterpiece , Photo, Super resolution , High Quality"
refine_neg = "(open mouth),[lowres],[smartphone camera], [amateur camera],[3d render],[sepia],((cartoon)),((anime)),((drawn)),(paint),(teeth), deformed, bad body proportions, mutation, (ugly), disfigured,(string)"
neg = "불완전한 신체 구조, 불명확, 잘림, 왜곡, 복사, 실수, 추가 팔, 추가 다리, 불쾌한 비율, 긴 목, 저급, 저해상도, 팔 다리 부족, 건강하지 않음, 유전적 변이, 프레임을 벗어남, 텍스트 삽입, 매력적이지 않음, 최저 품질"

# 라디오 버튼을 통해 선택할 수 있는 LoRA weights
lora_weights = {
    "기본 SD 1.5": None,
    "한복 LoRA": "/workspace/data/Lora_weights/koreandress_3",
    "수묵화 LoRA": "/workspace/data/Lora_weights/ad/checkpoint-2000"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        selected_lora = request.form["lora"]
       
        # LoRA weight 적용
        if selected_lora != "기본 SD 1.5":
            pipe.load_lora_weights(lora_weights[selected_lora])
        else:
            pipe.unload_lora_weights()
       
        # 이미지 생성
        image = pipe(prompt = prompt, negative_prompt = neg).images[0]
        refined = refiner(prompt = refine_prompt, negative_prompt = refine_neg , image = image).images[0]

        refined.save("generated_image.png")

        return send_file("generated_image.png", mimetype='image/png')
    return render_template("index.html", lora_weights=lora_weights.keys())

if __name__ == "__main__":
    app.run(debug=True)
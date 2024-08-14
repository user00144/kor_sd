import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from diffusers import StableDiffusionXLPipeline
from transformers import AutoModel
import torch
from conf.config import Config
from dataset.dataset import get_dataloader
import random
import numpy as np

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(CONFIG.SEED)

diff = StableDiffusionXLPipeline.from_pretrained(CONFIG.TEACHER_MODEL_NAME)#, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
student_model = AutoModel.from_pretrained(CONFIG.STUDENT_MODEL_NAME)#, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

print(student_model)

if CONFIG.TRAINING_TYPE == "default" :
    teacher_model = diff.text_encoder
else :
    teacher_model = diff.text_encoder_2

dl_train, dl_val = get_dataloader(CONFIG, diff)

from train import train

train(dl_train, dl_val, student_model, teacher_model, CONFIG)
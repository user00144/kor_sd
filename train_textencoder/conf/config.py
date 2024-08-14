import os

class Config () :
    ROOT_PATH = "/workspace/dev/ai_SD_finetuning/"
    DATA_PATH = "/workspace/data/text_data/text_data.csv"
    SEED = 42
    VAL_SIZE = 0.2
    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.001
    WD = 0.0001
    TRAINING_TYPE = "projection" # default , projection
    STUDENT_MODEL_NAME = "openai/clip-vit-large-patch14" if TRAINING_TYPE == "default" else "/workspace/dev/ai_SD_finetuning/archive/pretrained_projection_seed42"
    TOKENIZER_MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"
    TEACHER_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
    SAVE_DIR = os.path.join(ROOT_PATH, f"./archive/pretrained_{TRAINING_TYPE}_seed{SEED}")
    
import os

class Config () :
    ROOT_PATH = "/workspace/dev/segmind_vega/"
    DATA_PATH = "/workspace/data/text_data/text_data.csv"
    SEED = 42
    VAL_SIZE = 0.1
    BATCH_SIZE = 40
    EPOCHS = 30
    LR = 0.00001
    WD = 0.0001
    TRAINING_TYPE = "projection" # default , projection
    STUDENT_MODEL_NAME = "segmind/Segmind-Vega" if TRAINING_TYPE == "default" else "segmind/Segmind-Vega"
    TOKENIZER_MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"
    TEACHER_MODEL_NAME = "segmind/Segmind-Vega"
    SAVE_DIR = os.path.join(ROOT_PATH, f"./archive/pretrained_{TRAINING_TYPE}_seed{SEED}")
    
from safetensors.torch import load_file

# Anomal CLIP safetensors FILES
file_list = ['/workspace/data/models/archive/pretrained_default_seed42/model.safetensors']

full_state_dict = {}
for file in file_list :
    state_dict = load_file(file)
    for key, value in state_dict.items() :
        if key.startswith("text_model.module.") :
            new_key = "text_model." + key[len("text_model.module."):]
        else :
            new_key = key
        full_state_dict[new_key] = value


from transformers import CLIPModel, CLIPProcessor
# Normal CLIP Checkpoint
CLIP_PATH_2 = "/workspace/data/models/koclip_textmodel_1"

model = CLIPModel.from_pretrained(CLIP_PATH_2)
processor = CLIPProcessor.from_pretrained(CLIP_PATH_2)

model.load_state_dict(full_state_dict)


def find_key(dic, word) :
    matching_items = {}
    for key, value in dic.items() :
        if word in key :
            matching_items[key] = value
    return matching_items


print(model)

model.save_pretrained("./converted/")
processor.save_pretrained("./converted/")
import os
from PIL import Image
import random
from tqdm import tqdm

ROOT_PATH = "/workspace/data/korean_images/dataset"
DATA_PATH = os.path.join(ROOT_PATH, "TS_TS7")

list_file = os.listdir(DATA_PATH)
print('NUM_FILE : ', len(list_file))

def resize_and_random_crop_n(image_file_name, n):
    # 이미지 열기
    image = Image.open(os.path.join(DATA_PATH, image_file_name))
    width, height = image.size
    
    # 짧은 쪽을 512로 조정 (비율 유지)
    if width < height:
        new_width = 512
        new_height = int(height * (512 / width))
    else:
        new_height = 512
        new_width = int(width * (512 / height))
    
    # 이미지 크기 조정
    image = image.resize((new_width, new_height))
    
    # 크기 조정 후의 이미지 크기
    width, height = image.size
    
    # n개의 랜덤 크롭 생성
    for i in range(n):
        # 512x512 크기의 랜덤 위치 선택
        left = random.randint(0, width - 512)
        top = random.randint(0, height - 512)
        right = left + 512
        bottom = top + 512
        
        # 이미지 자르기
        cropped_image = image.crop((left, top, right, bottom))
        
        # 잘린 이미지 저장
        output_path = os.path.join(os.path.join(DATA_PATH, "crop"), f"{i}_{image_file_name}")
        cropped_image.save(output_path)
        
        
for image_file_name in tqdm(list_file) :
    try :
        resize_and_random_crop_n(image_file_name, 2)
    except :
        continue
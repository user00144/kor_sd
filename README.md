
# Building K-Contents Specialized Text2Image Pipeline
___



## Research on Building K-Contents Specialized Text2Image Pipeline
> **Jul. 2024 ~ Aug. 2024**

---

##  Team members
| **Name** | **Role** | **Email** |
|----------|----------|---------|
| **Seungeun Kang** | AI pipeline Development Leader | haun620@kyonggi.ac.kr |
| **Dahyun Lee**  | Data collection and preprocessing,  Lora training | idahyun22@kyonggi.ac.kr |
| **Sangbum Han** | Serving and Optimization pipeline | hsb422@kyonggi.ac.kr |

## Software Stacks
![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

---

## Project Motivation

### 1. Global interest in generative AI has increased
- Text2Image generative model is utilized in various fields and is used as an innovative tool in fields that require creativity

### 2. Recent "K-Contents Boom" has increased interest in Korean culture
- created a global demand for traditional Korean art, hanbok, Korean animation, etc...


**We were interested in building a pipeline that understands the Korean language and generates Korean-specific images in this technological and cultural context**

---

## Implementation

### 1. Collect Dataset and Preprocessing
- for text dataset, we used AI Hub Dataset(Korean English Translation Corpus Data - [Parallel](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=10&aihubDataSe=data&dataSetSn=126), [Technical Science](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=124), [Social Science](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=125))

- for image dataset, we used AI Hub Dataset([Korean traditional ink painting production data by style](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71380) , [Character Face Landmark Data in Animation](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=570) and Web Crawled Data

- Data Preprocessing

<img src="https://github.com/user-attachments/assets/ef9a9a70-4f65-4dcb-bf2d-2f0bf7874041" width=350 />  <img src="https://github.com/user-attachments/assets/45546b28-b817-4617-af9c-76990e65614a" width=400 />

### 2. Korean CLIP Text Encoder
- With BaseModel([Bingsu/clip-vit-large-patch14-ko](https://huggingface.co/Bingsu/clip-vit-large-patch14-ko)) ...
- We trained CLIP TextEncoder using (BaseModel, text datasets)

![image](https://github.com/user-attachments/assets/a4d5e3aa-0169-4a15-8bdc-6f25e408f5ee)

### 3. Pipeline Design
![362559460-e6190342-15ec-4a3e-9279-30a1588e9517](https://github.com/user-attachments/assets/e8b1b783-08b6-43d4-a19a-758d2895c210)

### 4. Experiments
<img src="https://github.com/user-attachments/assets/681f58a7-0635-437c-aac3-3d364c0d181e" width=800 />



---

## Outputs

- **Publication Conference Paper** in The 27th International Conference on Advanced Communication Technology, (Feb. 2025)
![image](https://github.com/user-attachments/assets/69e3e570-4c3e-44ee-ad78-d82de9f959e9)

- **Demo Web Page**
![image](https://github.com/user-attachments/assets/7276688e-86dd-4ecd-916a-5b886b944323)

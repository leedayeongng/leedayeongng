
import cv2
import numpy as np
import json
from pathlib import Path
import shutil

# 원본 → 작업폴더 복사
src = "/안심존데이터/초음파/"
dst = "/home/ubuntu/project/data/ultrasound/"
# shutil.copytree(src, dst)  # 마운트 후 주석 해제

img_paths = list(Path(dst).glob("**/*.png"))
print(f"총 초음파 이미지: {len(img_paths)}개")

def preprocess_ultrasound(img_path, size=(224, 224)):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img / 255.0  # 정규화
    return img

# JSON 라벨 읽기
def load_label(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

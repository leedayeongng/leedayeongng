
import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
from pathlib import Path

# ================================
# 1. 데이터 구조 파악
# ================================
data_root = "/안심존데이터/"

for root, dirs, files in os.walk(data_root):
    level = root.replace(data_root, "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 2:
        subindent = " " * 2 * (level + 1)
        for file in files[:5]:
            print(f"{subindent}{file}")

# ================================
# 2. EMR 메타데이터 분석
# ================================
def load_emr(path):
    import shutil
    shutil.copytree("/안심존데이터/EMR/", "/home/ubuntu/project/data/emr/")
    df = pd.read_csv(path)
    print(df.shape)
    print(df.dtypes)
    print(df.isnull().sum())
    print(df.describe())
    return df

# ================================
# 3. 초음파 이미지 로드
# ================================
def load_ultrasound(img_dir):
    img_paths = list(Path(img_dir).glob("**/*.dcm"))
    print(f"총 이미지 수: {len(img_paths)}")

    # DICOM 읽기
    ds = pydicom.dcmread(str(img_paths[0]))
    img = ds.pixel_array
    print(f"Shape: {img.shape}, dtype: {img.dtype}")
    return img_paths

# ================================
# 4. GPU 확인
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


import pandas as pd
import numpy as np
import shutil

# 원본 → 작업폴더 복사
src = "/안심존데이터/메타데이터/"
dst = "/home/ubuntu/project/data/metadata/"
# shutil.copytree(src, dst)  # 마운트 후 주석 해제

def load_metadata(path):
    df = pd.read_csv(path)  # 또는 read_excel
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())
    print("결측치:", df.isnull().sum())
    print(df.describe())
    return df

# 주요 컬럼 (난소암 기준)
# CA-125, 나이, 병기, 조직학적유형, 악성여부
def preprocess_emr(df):
    df = df.dropna(subset=["label"])  # 라벨 없는 행 제거
    df = pd.get_dummies(df, columns=["조직학적유형"])  # 범주형 인코딩
    return df

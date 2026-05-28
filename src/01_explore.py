
import os
import json
import pandas as pd
from pathlib import Path

data_root = "/안심존데이터/"  # 마운트 후 실제 경로 확인

# 폴더 구조 파악
for root, dirs, files in os.walk(data_root):
    level = root.replace(data_root, "").count(os.sep)
    if level > 2:
        continue
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files[:3]:
        print(f"{subindent}{file}")

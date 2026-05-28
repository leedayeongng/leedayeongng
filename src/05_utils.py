
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 불균형 데이터 처리
# ================================
from imblearn.over_sampling import SMOTE

def balance_data(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"Before: {len(y)} → After: {len(y_res)}")
    return X_res, y_res

# ================================
# 의료 AI 핵심 지표
# ================================
def evaluate(y_true, y_pred, y_prob):
    print(classification_report(y_true, y_pred,
          target_names=["양성", "악성"]))
    print(f"AUC: {roc_auc_score(y_true, y_prob):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"민감도(Sensitivity): {tp/(tp+fn):.4f}")
    print(f"특이도(Specificity): {tn/(tn+fp):.4f}")

# ================================
# GradCAM 시각화
# ================================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_gradcam(model, img_tensor, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor)
    return grayscale_cam

# ================================
# Early Stopping
# ================================
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ================================
# 학습 곡선 저장 (반출용)
# ================================
def plot_history(train_loss, val_loss, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)  # 반출 가능한 단순 선그래프
    print(f"저장완료: {save_path}")

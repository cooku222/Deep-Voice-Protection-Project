import os
import numpy as np
import torch
from torch.utils.data import Dataset

class VoicePhishingDataset(Dataset):
    def __init__(self, base_dir="data", target_len=16000):
        self.data = []
        self.labels = []
        self.target_len = target_len

        for label, folder in enumerate(['label0', 'label1']):
            path = os.path.join(base_dir, folder)
            if not os.path.exists(path):
                print(f"[경고] 폴더 없음: {path}")
                continue

            for file in os.listdir(path):
                if file.endswith(".npy"):
                    self.data.append(os.path.join(path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path = self.data[idx]
        x = np.load(npy_path)

        # ✅ 채널 처리
        if x.ndim == 1:
            x = np.stack([x, x], axis=0)  # mono → stereo
        elif x.ndim == 2:
            if x.shape[0] == 2:
                pass  # 정상
            elif x.shape[0] == 1:
                x = np.vstack([x, x])  # (1, N) → (2, N)
            elif x.shape[1] == 2:
                x = x.T  # (N, 2) → (2, N)
            else:
                raise ValueError(f"비정상 shape: {x.shape}, 파일: {npy_path}")
        else:
            raise ValueError(f"지원되지 않는 shape: {x.shape}, 파일: {npy_path}")

        # ✅ 길이 맞추기
        if x.shape[1] > self.target_len:
            x = x[:, :self.target_len]
        elif x.shape[1] < self.target_len:
            pad_len = self.target_len - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad_len)))

        # 정규화 수정 - 에너지 기준 RMS 스케일링
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-7)

        x = torch.tensor(x, dtype=torch.float32)  # shape: [2, 16000]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

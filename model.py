import torch.nn as nn
import torch

class StereoCNN(nn.Module):
    def __init__(self):
        super(StereoCNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=5, stride=2, padding=2)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.act3 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):  # x: [batch, 2, 16000]
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool(x)  # [batch, 128, 1]
        x = x.view(x.size(0), -1)  # [batch, 128]
        x = self.dropout(x)
        out = self.fc(x)

        # 🔍 디버깅 출력
        if not self.training:
            print("[DEBUG] Model output (logits):", out.detach().cpu().numpy())

        return out

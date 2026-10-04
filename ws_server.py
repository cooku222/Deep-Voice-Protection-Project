from fastapi import FastAPI, WebSocket
import numpy as np
import torch
from model import StereoCNN
from io import BytesIO

app = FastAPI()

# ✅ 모델 로드
model = StereoCNN()
model.load_state_dict(torch.load("model_stereo.pt", map_location="cpu"))
model.eval()

def preprocess(data_bytes, target_len=16000):
    x = np.load(BytesIO(data_bytes))

    # 채널 처리
    if x.ndim == 1:
        x = np.stack([x, x], axis=0)
    elif x.ndim == 2:
        if x.shape[0] == 1:
            x = np.vstack([x, x])
        elif x.shape[1] == 2:
            x = x.T
    else:
        raise ValueError(f"Invalid shape: {x.shape}")

    # 길이 맞추기
    if x.shape[1] > target_len:
        x = x[:, :target_len]
    elif x.shape[1] < target_len:
        x = np.pad(x, ((0, 0), (0, target_len - x.shape[1])))

    # 정규화
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-7)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, 2, 16000]
    return x

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()  # 클라이언트에서 바이너리로 npy 보내야 함
            x = preprocess(data)
            with torch.no_grad():
                output = model(x)
                pred = torch.argmax(output, dim=1).item()
            await websocket.send_json({"result": int(pred)})
        except Exception as e:
            await websocket.send_json({"error": str(e)})
            break

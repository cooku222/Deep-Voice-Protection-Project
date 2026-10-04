import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import VoicePhishingDataset
from model import StereoCNN
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

def calculate_metrics(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = 100 * correct / total

    # Precision, Recall
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')

    return accuracy, precision, recall

def main():
    # ✅ 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ✅ 데이터 로딩
    dataset = VoicePhishingDataset(base_dir="data")  # ← 폴더 경로 필요시 변경
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ✅ 모델
    model = StereoCNN().to(device)

    # ✅ 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5) 

    # ✅ 학습 루프
    for epoch in range(100):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
                                
            total_loss += loss.item()

        scheduler.step() 

        # ✅ 에폭 끝나고 전체 정확도, 정밀도, 재현율 계산
        accuracy, precision, recall = calculate_metrics(model, train_loader, device)

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # ✅ 모델 저장
    torch.save(model.state_dict(), "model_stereo.pt")
    print("Model saved to model_stereo.pt")

if __name__ == "__main__":
    main()

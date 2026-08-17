import torch
from ultralytics import YOLO

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=r"E:\EchoPath\dataset_merged_v3\dataset_merged_v3\data.yaml",
        epochs=100,
        imgsz=640,
        batch=64,
        patience=30,
        device=device,
        workers=12,
        project="runs/echopath",
        name="v3",
    )

if __name__ == "__main__":
    main()
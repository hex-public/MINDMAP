# scripts/train_cls.py
from ultralytics import YOLO
from pathlib import Path
import shutil

# ✅ 결과폴더를 명시적으로 runs/classify로 고정
PROJECT = Path("runs/classify")
RUN_NAME = "alz4_run"

DATA = Path("data/alz4_yolo")
IMG_SIZE, EPOCHS, BATCH = 224, 25, 32

def main():
    model = YOLO("yolov8n-cls.pt")
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(PROJECT),    # <-- 여기가 포인트
        name=RUN_NAME,           # runs/classify/alz4_run/
        verbose=False
    )

    # ✅ 저장 경로 일관화
    src = PROJECT / RUN_NAME / "weights" / "best.pt"     # runs/classify/alz4_run/weights/best.pt
    dst = Path("weights") / "best.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print("=> Saved:", dst.resolve())

if __name__ == "__main__":
    main()

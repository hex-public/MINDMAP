# scripts/prepare_dataset.py
from pathlib import Path
import shutil, os, random
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

CLASS_NAMES = ["NonDemented","VeryMildDemented","MildDemented","ModerateDemented"]
IMG_EXT = {".jpg",".jpeg",".png"}

RAW = Path("data/raw")            # 원본(압축 해제해둔 경로)
OUT = Path("data/alz4_yolo")      # YOLO 분류 구조로 정리될 경로
OUT.mkdir(parents=True, exist_ok=True)

def scan_images():
    paths, labels = [], []
    for root, _, files in os.walk(RAW):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXT:
                p = Path(root)/f
                label = Path(root).name
                label_norm = label.lower().replace(" ", "")
                mapping = {
                    "nondemented":"NonDemented",
                    "verymilddemented":"VeryMildDemented",
                    "milddemented":"MildDemented",
                    "moderatedemented":"ModerateDemented",
                }
                label_std = mapping.get(label_norm, label)
                if label_std in CLASS_NAMES:
                    paths.append(str(p))
                    labels.append(label_std)
    return pd.DataFrame({"path": paths, "label": labels})

def make_dirs():
    for split in ["train","val","test"]:
        for cls in CLASS_NAMES:
            (OUT/split/cls).mkdir(parents=True, exist_ok=True)

def copy_split(df, split):
    from tqdm import tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copy {split}"):
        src = Path(row["path"])
        dst = OUT/split/row["label"]/src.name
        if not dst.exists():
            shutil.copy2(src, dst)

def main():
    df = scan_images()
    print("총 이미지:", len(df))
    print(df["label"].value_counts())

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=(1/3), stratify=temp_df["label"], random_state=SEED)

    make_dirs()
    copy_split(train_df, "train")
    copy_split(val_df,   "val")
    copy_split(test_df,  "test")

    print("완료. OUT:", OUT.resolve())

if __name__ == "__main__":
    main()

# classifier.py
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

def load_model(weights_path: str | Path) -> YOLO:
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"가중치가 없습니다: {p.resolve()}")
    return YOLO(str(p))

def predict_image(model: YOLO, pil_image: Image.Image, imgsz: int = 224, topk: int = 3) -> Dict[str, Any]:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    res = model.predict(source=np.array(pil_image), imgsz=imgsz, verbose=False)[0]
    names = res.names
    probs = res.probs.data.cpu().numpy()
    idxs = probs.argsort()[::-1][:topk]
    topk_list = [{"label": names[int(i)], "conf": float(probs[int(i)]), "index": int(i)} for i in idxs]
    return {"top1": topk_list[0], "topk": topk_list, "names": names}
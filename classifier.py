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

def class_distribution(model: YOLO, pil_image: Image.Image, imgsz: int = 224):
    """전체 클래스별 확률 분포 반환 (Top-K 차트용)"""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    res = model.predict(source=np.array(pil_image), imgsz=imgsz, verbose=False)[0]
    names = res.names
    probs = res.probs.data.cpu().numpy()
    labels = [names[i] for i in range(len(probs))]
    return labels, (probs * 100).tolist()


#Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except Exception:
    HAS_GRADCAM = False


def make_gradcam(model: YOLO, pil_image: Image.Image, target_index: int = None, imgsz: int = 224):
    """Grad-CAM 히트맵 생성 (pytorch-grad-cam 설치 필요)"""
    if not HAS_GRADCAM:
        return None, None
    import torch
    import numpy as np

    from PIL import Image

    # torch 모델 내부 접근
    torch_model = model.model
    # 마지막 conv layer 찾기
    target_layer = None
    for m in torch_model.modules():
        if "Conv" in m.__class__.__name__:
            target_layer = m

    if target_layer is None:
        return None, None

    # 전처리
    img_rgb = pil_image.convert("RGB").resize((imgsz, imgsz))
    np_img = np.asarray(img_rgb).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        out = torch_model(tensor)
        probs = torch.softmax(out, dim=1)
        if target_index is None:
            target_index = int(torch.argmax(probs, dim=1).item())

    targets = [ClassifierOutputTarget(target_index)]
    cam = GradCAM(model=torch_model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    overlay = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)
    overlay_pil = Image.fromarray(overlay)
    heatmap_pil = Image.fromarray((grayscale_cam * 255).astype(np.uint8)).convert("L")

    return overlay_pil, heatmap_pil

import torch
from ultralytics import YOLO

def get_torch_model(yolo_model: YOLO) -> torch.nn.Module:
    """
    Ultralytics YOLO 분류 모델에서 실제 nn.Module을 꺼내 반환.
    Grad-CAM이 backward 가능한 형태로 접근할 수 있도록 내부 .model.model까지 탐색함.
    """
    # YOLO 모델의 내부 구조 접근
    inner = getattr(yolo_model, "model", None)
    if inner is None:
        raise RuntimeError("YOLO 내부 nn.Module을 찾을 수 없습니다.")

    # 분류 모델일 경우에는 inner.model이 실제 torch.Module
    core = getattr(inner, "model", None)
    if core is not None and isinstance(core, torch.nn.Module):
        return core

    # 아니라면 inner 자체가 nn.Module인 경우 반환
    if isinstance(inner, torch.nn.Module):
        return inner

    raise RuntimeError("YOLO 내부 nn.Module 접근에 실패했습니다.")
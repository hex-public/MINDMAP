# scripts/cam_cls.py
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _to_tensor(pil: Image.Image, size: int) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),                     # [0,1]
        # 분류 학습 때 정규화 썼다면 여기에도 동일 적용 (예: ImageNet)
        # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    x = t(pil.convert("RGB")).unsqueeze(0)        # [1,3,H,W]
    x.requires_grad_(True)                        # ✅ GradCAM 핵심
    return x


def _find_last_conv_layer(model: nn.Module) -> nn.Module:
    """모델에서 마지막 Conv2d 레이어를 찾아 Grad-CAM 타깃으로 사용."""
    target = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            target = m
    if target is None:
        raise RuntimeError("Conv2d layer를 찾지 못했습니다. 모델 구조를 확인하세요.")
    return target


def gradcam_overlay_for_cls(
    torch_model: nn.Module,
    pil_image: Image.Image,
    input_size: int = 224,
    target_index: Optional[int] = None,
    alpha: float = 0.45,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Returns:
      overlay_pil: CAM이 입혀진 PIL 이미지
      grayscale_cam: [H,W] CAM 배열 (0~1)
    """
    # Grad가 켜져 있어야 함
    torch.set_grad_enabled(True)
    torch_model.eval()

    # M2에서 MPS가 간혹 역전파 이슈를 내면 CPU로 강제
    device = next(torch_model.parameters()).device
    # device = torch.device("cpu")  # 필요 시 주석 해제

    x = _to_tensor(pil_image, input_size).to(device)

    # 타깃 레이어 선택
    target_layers = [_find_last_conv_layer(torch_model)]

    # 타깃 클래스 지정 (없으면 모델이 자동 선택)
    targets = [ClassifierOutputTarget(int(target_index))] if target_index is not None else None

    # 최신 버전: use_cuda 인자 없음
    cam = GradCAM(model=torch_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=x, targets=targets, eigen_smooth=True)  # [B,H,W]
    grayscale_cam = grayscale_cam[0]

    # 시각화
    img_np = np.array(pil_image.convert("RGB").resize((input_size, input_size)), dtype=np.float32) / 255.0
    vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=1 - alpha)
    return Image.fromarray(vis), grayscale_cam


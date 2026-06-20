"""Tien ich super-resolution (RRDBNet x4) - dung RRDBNet standalone (khong basicsr)."""
import cv2
import numpy as np
import torch

from rrdbnet_arch import RRDBNet


def load_sr_model(ckpt_path, device, scale=4):
    """Nap RRDBNet SR tu checkpoint. Ho tro key params_ema / params / model."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state, dict):
        for k in ("params_ema", "params", "model"):
            if k in state:
                state = state[k]
                break
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


@torch.no_grad()
def super_resolve(model, img_bgr, device):
    """Chay SR tren 1 anh BGR (uint8) -> anh BGR (uint8) da phong to x4."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    out = model(t).clamp(0, 1)
    out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

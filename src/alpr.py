"""
Pipeline ALPR bien so VN (ban dong goi - chay offline).

Luong: anh -> YOLOv5 detect bien -> crop -> Real-ESRGAN (SR) lam net
       -> deskew -> YOLOv5 doc ky tu.

Models trong models/ ; YOLOv5 nap OFFLINE tu vendor/yolov5 (khong can mang).
"""
from pathlib import Path
import sys

import cv2
import torch

SRC = Path(__file__).resolve().parent
PKG = SRC.parent
sys.path.insert(0, str(SRC))

import helper            # noqa: E402
import utils_rotate      # noqa: E402
from sr_model import load_sr_model, super_resolve   # noqa: E402

YOLOV5_DIR = PKG / "vendor" / "yolov5"
MODELS = PKG / "models"
DET_MODEL = MODELS / "LP_detector.pt"
OCR_MODEL = MODELS / "LP_ocr.pt"
SR_MODEL = MODELS / "best_plate_sr.pth"
OCR_CONF = 0.6
SCALE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_yolo(weights):
    """Nap YOLOv5 custom OFFLINE tu repo vendor (source='local')."""
    return torch.hub.load(str(YOLOV5_DIR), "custom", path=str(weights),
                          source="local", verbose=False)


def _clip_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def normalize(text):
    return "".join(c for c in (text or "").upper() if c.isalnum())


class ALPR:
    def __init__(self, use_sr=True, ocr_conf=OCR_CONF):
        self.use_sr = use_sr
        self.det = _load_yolo(DET_MODEL)
        self.ocr = _load_yolo(OCR_MODEL)
        self.ocr.conf = ocr_conf
        self.sr = load_sr_model(SR_MODEL, DEVICE, SCALE) if use_sr else None

    def detect_boxes(self, img):
        """Phat hien TAT CA bien -> list (xyxy, crop), sap theo trai->phai roi tren->duoi."""
        res = self.det(img, size=640)
        boxes = res.pandas().xyxy[0].values.tolist()
        out = []
        for b in boxes:
            crop = _clip_crop(img, b[0], b[1], b[2], b[3])
            if crop is not None:
                xyxy = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                out.append((xyxy, crop))
        out.sort(key=lambda t: (t[0][1] // 50, t[0][0]))  # gom theo hang roi trai->phai
        return out

    def detect_crop(self, img):
        """(Tuong thich cu) tra ve crop cua bien LON NHAT, hoac None."""
        boxes = self.detect_boxes(img)
        if not boxes:
            return None
        return max(boxes, key=lambda t: (t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))[1]

    def read_crop(self, crop):
        for cc in range(2):
            for ct in range(2):
                cand = helper.read_plate(self.ocr, utils_rotate.deskew(crop, cc, ct))
                if cand != "unknown":
                    return normalize(cand)
        return ""

    def _read_processed(self, crop):
        """SR (neu bat) roi doc -> (bien_so, anh_da_xu_ly)."""
        proc = super_resolve(self.sr, crop, DEVICE) if self.use_sr else crop
        return self.read_crop(proc), proc

    def predict(self, img_bgr, assume_cropped=False):
        """anh BGR -> (bien_so, crop_da_xu_ly) cho 1 bien (lon nhat).

        assume_cropped=True: anh dau vao DA la crop bien san -> bo qua buoc detect.
        """
        if assume_cropped:
            return self._read_processed(img_bgr)
        crop = self.detect_crop(img_bgr)
        if crop is None:
            return "", None
        return self._read_processed(crop)

    def predict_all(self, img_bgr, assume_cropped=False):
        """anh BGR -> list cac bien tim thay: [{box, text, proc}].

        Ho tro anh CO NHIEU bien so. assume_cropped=True: coi ca anh la 1 crop.
        """
        if assume_cropped:
            text, proc = self._read_processed(img_bgr)
            h, w = img_bgr.shape[:2]
            return [{"box": (0, 0, w, h), "text": text, "proc": proc}]
        results = []
        for xyxy, crop in self.detect_boxes(img_bgr):
            text, proc = self._read_processed(crop)
            results.append({"box": xyxy, "text": text, "proc": proc})
        return results

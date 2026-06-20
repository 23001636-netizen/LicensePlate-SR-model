"""
Demo web (Gradio): upload anh -> YOLOv5 detect+crop -> SR lam net -> YOLOv5 doc ky tu.
Hien crop goc vs sau SR + bien doc duoc (co/khong SR) de thay dong gop cua SR.

    python src/demo_web.py    -> mo http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import cv2
import gradio as gr

SRC = Path(__file__).resolve().parent
PKG = SRC.parent
sys.path.insert(0, str(SRC))
from alpr import ALPR, super_resolve, DEVICE   # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("[+] Dang nap mo hinh (lan dau hoi lau)...")
alpr = ALPR()
print("[+] San sang.")


def fmt_vn(s):
    if not s:
        return ""
    n = len(s)
    if n == 9:
        return f"{s[:2]}-{s[2:4]} {s[4:7]}.{s[7:]}"
    if n == 8:
        return f"{s[:2]}-{s[2:3]} {s[3:6]}.{s[6:]}"
    return s


def run(image_rgb, skip_detect):
    if image_rgb is None:
        return "Chua co anh", None, None, ""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if skip_detect:
        crop = img
    else:
        crop = alpr.detect_crop(img)
        if crop is None:
            return ("Khong detect duoc bien (neu anh da la crop bien san, "
                    "tich o ben trai)"), None, None, ""
    sr_crop = super_resolve(alpr.sr, crop, DEVICE)
    text_sr = alpr.read_crop(sr_crop)
    text_nosr = alpr.read_crop(crop)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    sr_rgb = cv2.cvtColor(sr_crop, cv2.COLOR_BGR2RGB)
    label = (f"OK  {text_sr}   ({fmt_vn(text_sr)})" if text_sr
             else "Khong doc duoc ky tu")
    note = (f"Doc khong SR: {text_nosr or '(khong doc duoc)'}\n"
            f"Doc CO SR  : {text_sr or '(khong doc duoc)'}")
    return label, crop_rgb, sr_rgb, note


examples = [[str(p)] for p in sorted((PKG / "samples").glob("*"))
            if p.suffix.lower() in EXTS][:12]

with gr.Blocks(title="ALPR bien so VN - SR + YOLOv5 OCR") as demo:
    gr.Markdown(
        "# Demo ALPR bien so xe Viet Nam\n"
        "**Luong:** anh -> YOLOv5 detect & crop -> **SR lam net** -> YOLOv5 doc ky tu")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Anh dau vao", type="numpy")
            skip = gr.Checkbox(label="Anh da la crop bien san (bo qua detect)", value=False)
            btn = gr.Button("Doc bien so", variant="primary")
        with gr.Column():
            out_text = gr.Label(label="Bien so")
            with gr.Row():
                out_crop = gr.Image(label="Crop goc (truoc SR)")
                out_sr = gr.Image(label="Sau SR (lam net)")
            out_note = gr.Textbox(label="Chi tiet", lines=2)
    btn.click(run, inputs=[inp, skip], outputs=[out_text, out_crop, out_sr, out_note])
    if examples:
        gr.Examples(examples=examples, inputs=inp)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

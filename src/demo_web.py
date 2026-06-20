"""
Demo web (Gradio): upload anh -> YOLOv5 detect+crop -> SR lam net -> YOLOv5 doc ky tu.
Ho tro anh CO NHIEU bien so: ve khung + bien doc duoc len anh, liet ke tat ca.

    python src/demo_web.py    -> mo http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

SRC = Path(__file__).resolve().parent
PKG = SRC.parent
sys.path.insert(0, str(SRC))
from alpr import ALPR, DEVICE   # noqa: E402

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


def _before_after(crop_bgr, sr_bgr):
    """Ghep [Truoc SR | Sau SR] cung chieu cao de thay SR lam net."""
    H = 160
    def fit(im):
        return cv2.resize(im, (max(1, int(im.shape[1] * H / im.shape[0])), H),
                          interpolation=cv2.INTER_NEAREST)
    a, b = fit(crop_bgr), fit(sr_bgr)
    gap = np.full((H, 10, 3), 255, np.uint8)
    panel = np.hstack([a, gap, b])
    panel = cv2.copyMakeBorder(panel, 26, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(panel, "Truoc SR", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.putText(panel, "Sau SR", (a.shape[1] + 22, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
    return cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)


def run(image_rgb, skip_detect):
    if image_rgb is None:
        return "Chua co anh", None, [], ""
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = alpr.predict_all(img, assume_cropped=skip_detect)
    if not results:
        return ("Khong detect duoc bien (neu anh da la crop bien san, tich o ben trai)",
                image_rgb, [], "")

    # ve khung + bien len anh
    vis = img.copy()
    h = img.shape[0]
    th = max(2, h // 400)
    fs = max(0.6, h / 1000)
    for i, r in enumerate(results, 1):
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), th)
        cv2.putText(vis, r["text"] or "?", (x1, max(int(20 * fs), y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 200, 0), th)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # gallery truoc/sau SR cho tung bien
    gallery = []
    for i, r in enumerate(results, 1):
        cap = f"Bien {i}: {r['text']}" if r["text"] else f"Bien {i}: (khong doc duoc)"
        gallery.append((_before_after(r["crop"], r["proc"]), cap))

    reads = [r["text"] for r in results if r["text"]]
    if len(results) == 1:
        t = results[0]["text"]
        label = f"OK  {t}   ({fmt_vn(t)})" if t else "Khong doc duoc ky tu"
    else:
        label = f"Tim thay {len(results)} bien so ({len(reads)} doc duoc)"
    lines = []
    for i, r in enumerate(results, 1):
        t = r["text"]
        lines.append(f"Bien {i}: {t + '  (' + fmt_vn(t) + ')' if t else '(khong doc duoc)'}")
    return label, vis_rgb, gallery, "\n".join(lines)


examples = [[str(p)] for p in sorted((PKG / "samples").glob("*"))
            if p.suffix.lower() in EXTS][:12]

with gr.Blocks(title="ALPR bien so VN - SR + YOLOv5 OCR") as demo:
    gr.Markdown(
        "# Demo ALPR bien so xe Viet Nam\n"
        "**Luong:** anh -> YOLOv5 detect & crop -> **SR lam net** -> YOLOv5 doc ky tu  \n"
        "Ho tro anh co **nhieu bien so**.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Anh dau vao", type="numpy")
            skip = gr.Checkbox(label="Anh da la crop bien san (bo qua detect)", value=False)
            gr.Markdown(
                ":warning: **Anh chup ca xe / khung canh thi DUNG tich o tren.** "
                "Chi tich khi anh da cat sat bien. Tich nham voi anh chua crop se "
                "khien demo doc sai hoac khong doc duoc.")
            btn = gr.Button("Doc bien so", variant="primary")
        with gr.Column():
            out_text = gr.Label(label="Ket qua")
            out_img = gr.Image(label="Anh ket qua (khung + bien so)")
            out_gallery = gr.Gallery(label="Tung bien: Truoc SR vs Sau SR",
                                     columns=2, height="auto")
            out_note = gr.Textbox(label="Danh sach bien", lines=4)
    btn.click(run, inputs=[inp, skip], outputs=[out_text, out_img, out_gallery, out_note])
    if examples:
        gr.Examples(examples=examples, inputs=inp)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

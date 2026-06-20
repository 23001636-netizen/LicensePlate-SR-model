# Demo ALPR + Super-Resolution — Nhận dạng biển số xe Việt Nam

Hệ thống nhận dạng biển số xe tự động (ALPR) có tích hợp **Siêu phân giải (SR)**
để làm nét vùng biển trước khi đọc, giúp đọc đúng cả biển mờ/chụp xa.

**Luồng xử lý:** ảnh → YOLOv5 phát hiện & cắt biển → Real-ESRGAN làm nét (×4) →
deskew → YOLOv5 đọc ký tự → ghép chuỗi biển số.

> Hỗ trợ ảnh có **nhiều biển số**: hệ thống phát hiện & đọc tất cả biển trong ảnh,
> vẽ khung + biển số đọc được lên ảnh kết quả.

---

## Cách chạy (3 bước)

> Yêu cầu: **Python 3.10 hoặc 3.11** đã cài sẵn, và **có Internet ở lần chạy đầu**
> (để tải thư viện về máy — chỉ lần đầu, các lần sau chạy ngay).

### Windows
1. Giải nén thư mục này ra ổ đĩa (ví dụ `D:\ALPR_SR_demo`).
2. **Double-click** `run_demo_windows.bat`.
3. Chờ cài thư viện (lần đầu ~vài phút). Khi xong, trình duyệt tự mở
   **http://127.0.0.1:7860** — giao diện demo.

### Linux / macOS
```bash
bash run_demo_linux_mac.sh
```
Rồi mở trình duyệt vào http://127.0.0.1:7860

---

## Dùng giao diện web
- Bấm vào một **ảnh mẫu** ở dưới (hoặc **kéo–thả ảnh** của bạn vào ô "Ảnh đầu vào").
- Bấm **"Đọc biển số"**. Kết quả hiển thị:
  - Biển số đọc được;
  - **Crop gốc** so với **Sau SR** (thấy rõ SR làm nét thế nào);
  - So sánh đọc *không SR* vs *có SR*.
- Nếu ảnh của bạn **đã là ảnh crop sát biển** (không phải ảnh chụp cả xe), hãy
  **tích ô "Ảnh đã là crop biển sẵn (bỏ qua detect)"** trước khi bấm đọc.

## Cách khác: chạy bằng dòng lệnh (không cần web)
Sau khi đã cài (chạy `run_demo` 1 lần), kích hoạt môi trường rồi:
```bash
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate

python src/demo_cli.py                 # đọc toàn bộ ảnh trong samples/
python src/demo_cli.py -i <duong_dan>  # ảnh hoặc thư mục bất kỳ
python src/demo_cli.py --no-sr         # tắt SR để so sánh
python src/demo_cli.py --cropped       # khi ảnh đã là crop biển sẵn
```
Ảnh kết quả (crop sau SR) được lưu trong thư mục `outputs/`.

---

## Chạy bằng GPU (tuỳ chọn, nhanh hơn)
Mặc định demo cài **PyTorch bản CPU** để chạy được mọi máy. Nếu máy có GPU NVIDIA
(CUDA 12.1), sau khi `run_demo` chạy 1 lần, cài lại torch bản GPU:
```bash
.venv\Scripts\activate            # (Windows)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```
Code tự động dùng GPU nếu phát hiện thấy.

---

## Cấu trúc thư mục
```
ALPR_SR_demo/
├─ run_demo_windows.bat      # chay tren Windows
├─ run_demo_linux_mac.sh     # chay tren Linux/macOS
├─ requirements.txt
├─ README.md
├─ src/
│  ├─ alpr.py                # pipeline: detect -> SR -> OCR
│  ├─ demo_web.py            # giao dien web (Gradio)
│  ├─ demo_cli.py            # chay dong lenh
│  ├─ sr_model.py            # nap & chay mo hinh SR
│  ├─ rrdbnet_arch.py        # kien truc RRDBNet (standalone)
│  ├─ helper.py / utils_rotate.py
├─ models/
│  ├─ LP_detector.pt         # YOLOv5 phat hien bien
│  ├─ LP_ocr.pt              # YOLOv5 doc ky tu (da fine-tune)
│  └─ best_plate_sr.pth      # Real-ESRGAN (SR) da fine-tune
├─ vendor/yolov5/            # ma nguon YOLOv5 (de chay OFFLINE)
└─ samples/                  # mot so anh thu nghiem
```

Demo **chạy hoàn toàn offline** (YOLOv5 nạp từ `vendor/yolov5`), không cần kết nối
mạng sau khi đã cài thư viện.

## Kết quả chính (tập ảnh thật)
| Tiền xử lý | Đọc đúng / 21 | Exact | Char acc |
|---|---|---|---|
| Không SR | 13 | 61.9% | 73.0% |
| Bicubic ×4 | 12 | 57.1% | 80.4% |
| **SR (mô hình này)** | **16** | **76.2%** | **87.3%** |

---
*Đồ án: Nâng cao độ phân giải ảnh biển số xe bằng Super-Resolution và ứng dụng vào ALPR.*

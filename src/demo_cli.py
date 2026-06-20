"""
Demo dong lenh (KHONG can server): doc bien tren cac anh trong thu muc.

    python src/demo_cli.py                 # chay tren samples/
    python src/demo_cli.py -i duong_dan    # anh hoac thu muc bat ky
    python src/demo_cli.py --no-sr         # tat SR de so sanh

Voi moi anh: in bien doc duoc va luu anh crop sau SR vao outputs/.
"""
import argparse
import sys
from pathlib import Path

import cv2

SRC = Path(__file__).resolve().parent
PKG = SRC.parent
sys.path.insert(0, str(SRC))
from alpr import ALPR, DEVICE   # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def fmt_vn(s):
    n = len(s)
    if n == 9:
        return f"{s[:2]}-{s[2:4]} {s[4:7]}.{s[7:]}"
    if n == 8:
        return f"{s[:2]}-{s[2:3]} {s[3:6]}.{s[6:]}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default=str(PKG / "samples"),
                    help="anh hoac thu muc anh (mac dinh samples/)")
    ap.add_argument("--no-sr", action="store_true", help="tat SR (doc thang)")
    ap.add_argument("--cropped", action="store_true",
                    help="anh dau vao DA la crop bien san -> bo qua buoc detect")
    ap.add_argument("--out", default=str(PKG / "outputs"), help="thu muc luu ket qua")
    args = ap.parse_args()

    inp = Path(args.input)
    imgs = ([inp] if inp.is_file()
            else sorted(p for p in inp.iterdir() if p.suffix.lower() in EXTS))
    if not imgs:
        print(f"[!] Khong co anh trong: {inp}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Thiet bi: {DEVICE} | SR: {'TAT' if args.no_sr else 'BAT'} | {len(imgs)} anh")
    print("[+] Dang nap mo hinh (lan dau hoi lau)...")
    alpr = ALPR(use_sr=not args.no_sr)
    print("-" * 56)

    n_ok = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  {p.name:<28} [loi doc anh]")
            continue
        text, crop = alpr.predict(img, assume_cropped=args.cropped)
        if text:
            n_ok += 1
            print(f"  {p.name:<28} -> {text}   ({fmt_vn(text)})")
        else:
            print(f"  {p.name:<28} -> (khong doc duoc)")
        if crop is not None:
            cv2.imwrite(str(out_dir / f"{p.stem}_out.png"), crop)
    print("-" * 56)
    print(f"[OK] Doc duoc {n_ok}/{len(imgs)} anh. Anh ket qua luu o: {out_dir}")


if __name__ == "__main__":
    main()

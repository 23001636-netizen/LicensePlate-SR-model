"""
Demo dong lenh (KHONG can server): doc bien tren cac anh trong thu muc.

    python src/demo_cli.py                 # chay tren samples/
    python src/demo_cli.py -i duong_dan    # anh hoac thu muc bat ky
    python src/demo_cli.py --no-sr         # tat SR de so sanh
    python src/demo_cli.py --cropped       # anh da la crop bien san

Ho tro anh CO NHIEU bien so. Voi moi anh: in cac bien doc duoc va luu anh
co ve khung + bien so vao outputs/.
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

    n_plates = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  {p.name:<28} [loi doc anh]")
            continue
        results = alpr.predict_all(img, assume_cropped=args.cropped)
        reads = [r for r in results if r["text"]]
        n_plates += len(reads)
        if not results:
            print(f"  {p.name:<28} -> (khong detect duoc bien)")
        elif len(results) == 1:
            t = results[0]["text"]
            print(f"  {p.name:<28} -> {t or '(khong doc duoc)'}"
                  + (f"   ({fmt_vn(t)})" if t else ""))
        else:
            # nhieu bien tren 1 anh
            strs = [f"{r['text']}" if r["text"] else "(?)" for r in results]
            print(f"  {p.name:<28} -> {len(results)} bien: " + ", ".join(strs))
        # ve khung + chu len anh goc roi luu
        vis = img.copy()
        for r in results:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(vis, r["text"] or "?", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.imwrite(str(out_dir / f"{p.stem}_out.png"), vis)
    print("-" * 56)
    print(f"[OK] Doc duoc {n_plates} bien tren {len(imgs)} anh. Anh ket qua (co khung) o: {out_dir}")


if __name__ == "__main__":
    main()

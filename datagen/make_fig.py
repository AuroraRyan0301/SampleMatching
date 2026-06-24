#!/usr/bin/env python3
"""Build the side-by-side method-comparison figure used in the README.

For one preview view of a scene, lay out
    [reference | initial | each method's final render]
in a single row, tonemap the linear HDR preview EXRs to sRGB, and annotate every
cell with its PSNR against the reference.

It reads the preview renders of an existing optimization run from
``$POSTTRACKING_OUTPUT_DIR`` (default ``<repo>/output``); run ``reproduce.py``
first. Requires ``opencv-python`` built with OpenEXR support.

Usage:
    python datagen/make_fig.py --config <config-name> [--view S] [--out PATH]
"""
import os, glob, math, argparse
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
OUTPUT_DIR = os.environ.get("POSTTRACKING_OUTPUT_DIR", os.path.join(REPO, "output"))

# (folder, title, subtitle); folder=None -> reference, "__init__" -> initial render
METHODS = [
    (None,                            "Reference",              ""),
    ("__init__",                      "Initial",                "(start)"),
    ("volpathsimple-drt-mis-n4",      "DRT (baseline)",         "quadratic"),
    ("volpathsimple-drt-mis-linear",  "DRT (baseline)",         "linear"),
    ("volpathfm-drt-sd-n4",           "Ours (Sample Matching)", "quadratic"),
    ("volpathfm-linear-drt-sd-n4",    "Ours (Sample Matching)", "linear"),
]
OURS, BASE_C = (20, 110, 200), (90, 90, 90)


def font(size, bold=False):
    suffix = "-Bold" if bold else ""
    for c in (f"/usr/share/fonts/TTF/DejaVuSans{suffix}.ttf",
              f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf"):
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img > limit, 1.055 * np.power(np.clip(img, 0, None), 1 / 2.4) - 0.055,
                   12.92 * img)
    return np.clip(img, 0, 1)


def load_srgb(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    a = cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2RGB)
    return (linear_to_srgb(a) * 255 + 0.5).astype(np.uint8)


def psnr(gt_path, r_path, mx=1.0):
    g = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float64)[..., :3]
    x = cv2.imread(r_path, cv2.IMREAD_UNCHANGED).astype(np.float64)[..., :3]
    mse = np.mean((g - x) ** 2)
    return float("inf") if mse == 0 else 20 * math.log10(mx / math.sqrt(mse))


def main():
    ap = argparse.ArgumentParser(description="Make the README comparison figure.")
    ap.add_argument("--config", default="bunny-cloud-l1-6e-3-formal-local-single-gpu",
                    help="optimization config name (a sub-dir of the output dir)")
    ap.add_argument("--view", type=int, default=0, help="preview sensor index to show")
    ap.add_argument("--out", default=os.path.join(REPO, "figures", "four_method_comparison.png"))
    args = ap.parse_args()

    base = os.path.join(OUTPUT_DIR, args.config)
    any_dir = os.path.join(base, "volpathfm-drt-sd-n4")
    s = args.view
    refp = os.path.join(any_dir, f"ref_{s:04d}.exr")
    if not os.path.isfile(refp):
        raise SystemExit(f"reference not found: {refp}\nRun reproduce.py for '{args.config}' first.")

    def cell(folder):
        if folder is None:
            return load_srgb(refp), None
        d = any_dir if folder == "__init__" else os.path.join(base, folder)
        tag = "init" if folder == "__init__" else "final"
        hits = glob.glob(os.path.join(d, f"opt_{tag}_{s:04d}_spp_*.exr"))
        if not hits:
            return None, None
        return load_srgb(hits[0]), psnr(refp, hits[0])

    CELL_W, PAD, HEADER_H, LABEL_H, ROWLBL_W = 300, 10, 64, 30, 70
    r0 = load_srgb(refp); h, w = r0.shape[:2]
    CELL_H = int(round(CELL_W * h / w))
    ncol = len(METHODS)
    FIG_W = ROWLBL_W + ncol * CELL_W + (ncol + 1) * PAD
    FIG_H = HEADER_H + (CELL_H + LABEL_H + PAD) + PAD + 28

    canvas = Image.new("RGB", (FIG_W, FIG_H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_title, f_sub = font(19, True), font(15)
    f_psnr, f_row = font(18, True), font(15, True)

    for j, (folder, title, sub) in enumerate(METHODS):
        x0 = ROWLBL_W + PAD + j * (CELL_W + PAD); cx = x0 + CELL_W // 2
        col = OURS if title.startswith("Ours") else (BASE_C if title.startswith("DRT") else (0, 0, 0))
        draw.text((cx, 6), title, font=f_title, fill=col, anchor="ma")
        if sub:
            draw.text((cx, 30), sub, font=f_sub, fill=col, anchor="ma")

    y0 = HEADER_H + PAD
    draw.text((ROWLBL_W // 2, y0 + CELL_H // 2), f"view {s}", font=f_row, fill=(0, 0, 0), anchor="mm")
    for j, (folder, title, sub) in enumerate(METHODS):
        x0 = ROWLBL_W + PAD + j * (CELL_W + PAD); cx = x0 + CELL_W // 2
        img, p = cell(folder)
        if img is not None:
            canvas.paste(Image.fromarray(img).resize((CELL_W, CELL_H), Image.LANCZOS), (x0, y0))
        is_ours = title.startswith("Ours")
        draw.rectangle([x0 - 1, y0 - 1, x0 + CELL_W, y0 + CELL_H],
                       outline=OURS if is_ours else (200, 200, 200), width=3 if is_ours else 1)
        if p is None:
            draw.text((cx, y0 + CELL_H + 5), "(ground truth)" if folder is None else "(missing)",
                      font=f_sub, fill=(0, 0, 0), anchor="ma")
        else:
            draw.text((cx, y0 + CELL_H + 4), f"{p:.2f} dB", font=f_psnr,
                      fill=OURS if is_ours else (0, 0, 0), anchor="ma")

    cap = (f"{args.config}  -  view {s}  -  PSNR vs reference "
           "(linear; sRGB-tonemapped for display)")
    draw.text((FIG_W // 2, FIG_H - 22), cap, font=f_sub, fill=(60, 60, 60), anchor="ma")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print("saved", args.out, canvas.size)


if __name__ == "__main__":
    main()

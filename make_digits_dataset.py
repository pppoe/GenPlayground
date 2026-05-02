"""
make_digits_dataset.py — render digits 0-9 in system fonts → 2D point clouds.

Output: digit_fonts_pc.pt  (same format as mnist_2d_pc.pt: keys 'data', 'labels')

Usage:
    python make_digits_dataset.py
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def discover_fonts(dirs=("/usr/share/fonts", "/usr/local/share/fonts")) -> list[str]:
    paths = []
    for d in dirs:
        for ext in ("*.ttf", "*.otf"):
            paths.extend(str(p) for p in Path(d).rglob(ext) if p.exists())
    # Also try matplotlib if available (cross-platform bonus)
    try:
        from matplotlib import font_manager as fm
        paths += [f.fname for f in fm.fontManager.ttflist if Path(f.fname).exists()]
    except Exception:
        pass
    # Deduplicate preserving order
    seen, result = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def render_digit(digit: int, font: ImageFont.FreeTypeFont, image_size: int) -> Image.Image:
    img = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(img)
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (image_size - w) // 2 - bbox[0]
    y = (image_size - h) // 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)
    return img


def image_to_point_cloud(
    img: Image.Image,
    num_points: int = 128,
    threshold: float = 0.5,
    noise_std: float = 0.01,
) -> torch.Tensor | None:
    arr = torch.tensor(np.array(img)).float() / 255.0
    coords = (arr > threshold).nonzero().float()   # (K, 2) row-col
    if len(coords) < 4:                            # skip nearly-blank renders
        return None
    coords = coords[:, [1, 0]] / (img.size[0] - 1)  # -> x-y, normalized [0,1]

    if len(coords) >= num_points:
        idx = torch.randperm(len(coords))[:num_points]
    else:
        idx = torch.randint(0, len(coords), (num_points,))
    points = coords[idx]
    points += torch.randn_like(points) * noise_std
    return points


def make_digits_dataset(
    image_size: int = 64,
    font_size: int = 52,
    num_points: int = 128,
    repeats_per_font: int = 5,   # independent point-cloud draws per (digit, font) pair
    threshold: float = 0.5,
    noise_std: float = 0.01,
    output_path: str = "digit_fonts_pc.pt",
):
    font_paths = discover_fonts()
    print(f"Discovered {len(font_paths)} font files")

    # Load fonts, skip any that PIL can't open
    fonts: list[tuple[str, ImageFont.FreeTypeFont]] = []
    for path in font_paths:
        try:
            fonts.append((path, ImageFont.truetype(path, font_size)))
        except Exception:
            pass
    print(f"Loaded {len(fonts)} fonts")

    pc_data, labels = [], []
    for digit in range(10):
        count = 0
        for font_path, font in fonts:
            img = render_digit(digit, font, image_size)
            # Check the render actually produced a visible glyph
            if np.array(img).sum() < 500:
                continue
            for _ in range(repeats_per_font):
                pts = image_to_point_cloud(img, num_points, threshold, noise_std)
                if pts is not None:
                    pc_data.append(pts)
                    labels.append(digit)
                    count += 1
        print(f"  digit {digit}: {count} samples from {count // repeats_per_font} fonts")

    data = torch.stack(pc_data)
    labels_t = torch.tensor(labels)
    torch.save({"data": data, "labels": labels_t}, output_path)
    print(f"\nSaved {output_path}  —  shape={tuple(data.shape)}, labels={tuple(labels_t.shape)}")


if __name__ == "__main__":
    make_digits_dataset()

import torch
import numpy as np
from PIL import Image, ImageDraw

checkpoint = torch.load('mnist_2d_pc.pt')
dataset = checkpoint['data']

def render_image(ts_pts, image_size, radius=2):
    pts = ts_pts.float().numpy()
    pts = np.clip(pts, 0, 1)
    pts = (pts * (image_size - 1)).astype(int)

    img = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(img)
    for x, y in pts:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)
    return img

def make_grid(cell, gap, cols, rows, images):
    grid = Image.new("L", (cols * (cell + gap) - gap, rows * (cell + gap) - gap), 0)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * (cell + gap), r * (cell + gap)))
    return grid

cell, gap, cols, rows = 128, 4, 4, 4
idx = torch.randint(dataset.shape[0], [cols * rows])
images = [render_image(dataset[sample_idx], cell) for sample_idx in idx]
grid = make_grid(cell, gap, cols, rows, images)
grid.save("data/tensor_vis.jpg")
print(f"Saved data/tensor_vis.jpg  ({grid.width}x{grid.height})")

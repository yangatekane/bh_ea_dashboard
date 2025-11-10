# processing/contour_report.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

def generate_contour_report(input_image_path: str, output_path: str):
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image {input_image_path}")

    img = cv2.normalize(img.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img_blur = cv2.GaussianBlur(img, (7, 7), 0)

    contours = measure.find_contours(img_blur, level=0.5)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_blur, cmap="viridis", origin="lower")
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], linewidth=1.3, color="white", alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Resistivity / Yield Index", rotation=270, labelpad=14)
    ax.set_title("AI Studio Contour Report", fontsize=14, pad=10)
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return output_path

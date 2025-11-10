import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import os

def generate_contour_report(input_image_path, output_path):
    """
    Generate an annotated contour-style report from a given image.
    - Detects zones of similar intensity (proxy for resistivity/yield)
    - Overlays contour lines and annotations
    - Saves a new image ready for AI interpretation (Gemma, etc.)
    """

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    # --- Load and preprocess ---
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image {input_image_path}")

    # Normalize to [0,1]
    img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Smooth a bit for cleaner contours
    img_blur = cv2.GaussianBlur(img, (7, 7), 0)

    # --- Generate contour levels ---
    levels = np.linspace(img_blur.min(), img_blur.max(), 10)
    contours = measure.find_contours(img_blur, level=0.5)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_blur, cmap="viridis", origin="lower")

    # Overlay contour lines
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], linewidth=1.5, color='white', alpha=0.8)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Resistivity / Yield Index", rotation=270, labelpad=14)

    # --- Annotate representative zones ---
    ax.text(20, 30, "🟩 High Yield Zone", color="white", fontsize=10,
            bbox=dict(facecolor="green", alpha=0.5, boxstyle="round,pad=0.3"))
    ax.text(20, img.shape[0]-40, "🟥 Low Resistivity Zone", color="white", fontsize=10,
            bbox=dict(facecolor="red", alpha=0.5, boxstyle="round,pad=0.3"))

    ax.set_title("AI Studio Contour Report", fontsize=14, pad=10)
    ax.axis("off")

    # --- Save output ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

    return output_path

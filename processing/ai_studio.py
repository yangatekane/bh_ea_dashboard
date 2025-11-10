import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import os, json
from datetime import datetime

def generate_contour_report(input_image_path, output_path):
    """
    Generate an annotated contour-style report from a given image.
    - Detects zones of similar intensity (proxy for resistivity/yield)
    - Overlays contour lines and annotations
    - Saves a new image and a JSON metadata file (for Gemma/AI context)
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

    # --- Metadata structure ---
    metadata = {
        "input_file": os.path.basename(input_image_path),
        "timestamp_utc": datetime.now().isoformat() + "Z",
        "num_contours": len(contours),
        "mean_intensity": float(np.mean(img_blur)),
        "max_intensity": float(np.max(img_blur)),
        "min_intensity": float(np.min(img_blur)),
        "contours": []
    }

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_blur, cmap="viridis", origin="lower")

    for i, c in enumerate(contours):
        ax.plot(c[:, 1], c[:, 0], linewidth=1.5, color='white', alpha=0.8)

        # Add to metadata
        y_min, y_max = float(np.min(c[:, 0])), float(np.max(c[:, 0]))
        x_min, x_max = float(np.min(c[:, 1])), float(np.max(c[:, 1]))
        cx, cy = float(np.mean(c[:, 1])), float(np.mean(c[:, 0]))

        metadata["contours"].append({
            "id": i + 1,
            "bbox": [x_min, y_min, x_max, y_max],
            "centroid": [cx, cy],
            "approx_area_px2": float((x_max - x_min) * (y_max - y_min))
        })

    # --- Add colorbar legend ---
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

    # --- Save output image ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

    # --- Save JSON metadata next to the image ---
    metadata_path = os.path.splitext(output_path)[0] + "_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved contour report: {output_path}")
    print(f"🧠 Saved metadata: {metadata_path}")

    return output_path, metadata_path

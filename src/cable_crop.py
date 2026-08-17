import cv2
import numpy as np


# ============================================================
# CREATE FULL-RESOLUTION MASK
# ============================================================

def polygon_to_mask(
    image_shape,
    polygon
):
    """
    Convert YOLO polygon coordinates into a binary mask
    having EXACTLY the same dimensions as the original image.

    Output:
        0   = background
        255 = cable
    """

    height, width = image_shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if polygon is None:

        return mask

    if len(polygon) < 3:

        return mask

    points = np.asarray(
        polygon,
        dtype=np.float32
    )

    # Round polygon coordinates
    points = np.round(
        points
    ).astype(np.int32)

    # Safety clipping
    points[:, 0] = np.clip(
        points[:, 0],
        0,
        width - 1
    )

    points[:, 1] = np.clip(
        points[:, 1],
        0,
        height - 1
    )

    cv2.fillPoly(
        mask,
        [points],
        255
    )

    return mask


# ============================================================
# PADDED BBOX
# ============================================================

def get_padded_bbox(
    bbox,
    image_shape,
    padding=250
):
    """
    Calculate the crop rectangle.

    The same coordinates are used for:
        1. RGB image
        2. segmentation mask

    This guarantees alignment.
    """

    height, width = image_shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = int(
        round(x1 - padding)
    )

    y1 = int(
        round(y1 - padding)
    )

    x2 = int(
        round(x2 + padding)
    )

    y2 = int(
        round(y2 + padding)
    )

    # Clip to original image
    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        width,
        x2
    )

    y2 = min(
        height,
        y2
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# MASK OVERLAY
# ============================================================

def create_mask_overlay(
    crop,
    mask_crop,
    alpha=0.45
):
    """
    Create a visualization showing the predicted
    cable mask on top of the original crop.

    This is ONLY for visualization.
    """

    if crop.shape[:2] != mask_crop.shape[:2]:

        raise ValueError(
            "\nCrop and mask dimensions do not match:\n"
            f"Crop: {crop.shape[1]} × {crop.shape[0]}\n"
            f"Mask: {mask_crop.shape[1]} × {mask_crop.shape[0]}"
        )

    overlay = crop.copy()

    mask_pixels = (
        mask_crop > 127
    )

    mask_color = np.zeros_like(
        crop
    )

    # Green mask visualization
    mask_color[:, :] = (
        0,
        255,
        0
    )

    if np.any(mask_pixels):

        original_pixels = (
            overlay[mask_pixels]
            .astype(np.float32)
        )

        green_pixels = (
            mask_color[mask_pixels]
            .astype(np.float32)
        )

        blended = (
            (1 - alpha)
            * original_pixels
            +
            alpha
            * green_pixels
        )

        overlay[mask_pixels] = (
            blended
            .astype(np.uint8)
        )

    return overlay


# ============================================================
# CROP CABLE
# ============================================================

def crop_cable(
    image,
    polygon,
    bbox,
    padding=250
):
    """
    Create:

        1. high-resolution RGB cable crop
        2. perfectly aligned binary cable mask
        3. padded bbox

    IMPORTANT:
        The RGB crop and mask crop are created using
        EXACTLY the same coordinates.
    """

    if image is None:

        raise ValueError(
            "Input image is None."
        )

    image_height, image_width = image.shape[:2]

    # ========================================================
    # FULL-RESOLUTION MASK
    # ========================================================

    full_mask = polygon_to_mask(
        image.shape,
        polygon
    )

    # ========================================================
    # VERIFY FULL MASK
    # ========================================================

    if full_mask.shape != (
        image_height,
        image_width
    ):

        raise RuntimeError(
            "\nFULL MASK SIZE ERROR\n"
            f"Image: "
            f"{image_width} × {image_height}\n"
            f"Mask: "
            f"{full_mask.shape[1]} × "
            f"{full_mask.shape[0]}"
        )

    # ========================================================
    # CROP COORDINATES
    # ========================================================

    padded_bbox = get_padded_bbox(
        bbox,
        image.shape,
        padding
    )

    x1, y1, x2, y2 = padded_bbox

    if x2 <= x1 or y2 <= y1:

        raise ValueError(
            f"Invalid bbox: {padded_bbox}"
        )

    # ========================================================
    # HIGH-RESOLUTION RGB CROP
    # ========================================================

    crop = image[
        y1:y2,
        x1:x2
    ].copy()

    # ========================================================
    # EXACT SAME CROP ON MASK
    # ========================================================

    mask_crop = full_mask[
        y1:y2,
        x1:x2
    ].copy()

    # ========================================================
    # GUARANTEE ALIGNMENT
    # ========================================================

    if crop.shape[:2] != mask_crop.shape[:2]:

        raise RuntimeError(
            "\nCROP/MASK ALIGNMENT ERROR\n"
            f"BBox: {padded_bbox}\n"
            f"Crop: "
            f"{crop.shape[1]} × {crop.shape[0]}\n"
            f"Mask: "
            f"{mask_crop.shape[1]} × "
            f"{mask_crop.shape[0]}"
        )

    # ========================================================
    # FORCE BINARY MASK
    # ========================================================

    mask_crop = np.where(
        mask_crop > 127,
        255,
        0
    ).astype(np.uint8)

    # ========================================================
    # MASK QUALITY CHECK
    # ========================================================

    cable_pixels = np.count_nonzero(
        mask_crop
    )

    total_pixels = mask_crop.size

    coverage = (
        100.0
        * cable_pixels
        / total_pixels
        if total_pixels > 0
        else 0
    )

    print(
        f"  Mask coverage: "
        f"{coverage:.2f}%"
    )

    if cable_pixels == 0:

        print(
            "  ⚠️ WARNING: EMPTY CABLE MASK"
        )

    # ========================================================
    # RETURN
    # ========================================================

    return (
        crop,
        mask_crop,
        padded_bbox
    )
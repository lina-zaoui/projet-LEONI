import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MODEL1_ROOT = Path(
    r"C:\Users\Administrator\Desktop\stage-LEONI\projet-LEONI\data\output_model1\valid_output"
)

CROPS_DIR = (
    MODEL1_ROOT
    /
    "crops"
)

MASKS_DIR = (
    MODEL1_ROOT
    /
    "masks"
)

RESULTS_JSON = (
    MODEL1_ROOT
    /
    "results"
    /
    "model1_results.json"
)


MODEL2_ROOT = Path(
    r"C:\Users\Administrator\Desktop\stage-LEONI\projet-LEONI\data\model2_dataset"
)


# Model 2 training size
IMG_SIZE = 640


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

MODEL2_IMAGES = (
    MODEL2_ROOT
    /
    "images"
)

MODEL2_MASKS = (
    MODEL2_ROOT
    /
    "masks"
)

MODEL2_PREVIEWS = (
    MODEL2_ROOT
    /
    "previews"
)


MODEL2_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)

MODEL2_MASKS.mkdir(
    parents=True,
    exist_ok=True
)

MODEL2_PREVIEWS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LETTERBOX
# ============================================================

def letterbox_white(
    image,
    target_size=640,
    is_mask=False
):
    """
    Resize while preserving aspect ratio.

    Background is white.

    For masks:
        interpolation = NEAREST

    For RGB images:
        interpolation = INTER_AREA
    """

    h, w = image.shape[:2]

    if h <= 0 or w <= 0:

        raise ValueError(
            "Invalid image dimensions."
        )

    scale = min(
        target_size / w,
        target_size / h
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    if is_mask:

        interpolation = cv2.INTER_NEAREST

    else:

        interpolation = cv2.INTER_AREA

    resized = cv2.resize(
        image,
        (
            new_w,
            new_h
        ),
        interpolation=interpolation
    )

    # ========================================================
    # WHITE CANVAS
    # ========================================================

    if len(image.shape) == 2:

        canvas = np.full(
            (
                target_size,
                target_size
            ),
            255,
            dtype=np.uint8
        )

    else:

        canvas = np.full(
            (
                target_size,
                target_size,
                3
            ),
            255,
            dtype=np.uint8
        )

    x_offset = (
        target_size
        -
        new_w
    ) // 2

    y_offset = (
        target_size
        -
        new_h
    ) // 2

    canvas[
        y_offset:
        y_offset + new_h,
        x_offset:
        x_offset + new_w
    ] = resized

    return canvas


# ============================================================
# PROCESS ONE CABLE
# ============================================================

def process_cable(
    cable_info,
    image_name
):

    cable_id = cable_info[
        "cable_id"
    ]

    # ========================================================
    # GET MODEL 1 FILE PATHS
    # ========================================================

    crop_path = Path(
        cable_info["crop"]
    )

    mask_path = Path(
        cable_info["mask"]
    )

    # ========================================================
    # FALLBACK TO MODEL 1 DIRECTORIES
    # ========================================================

    if not crop_path.exists():

        crop_path = (
            CROPS_DIR
            /
            crop_path.name
        )

    if not mask_path.exists():

        mask_path = (
            MASKS_DIR
            /
            mask_path.name
        )

    # ========================================================
    # CHECK CROP
    # ========================================================

    if not crop_path.exists():

        print(
            f"  ❌ Crop not found:"
            f" {crop_path}"
        )

        return False

    # ========================================================
    # CHECK MASK
    # ========================================================

    if not mask_path.exists():

        print(
            f"  ❌ Mask not found:"
            f" {mask_path}"
        )

        return False

    # ========================================================
    # LOAD HIGH-RESOLUTION RGB CROP
    # ========================================================

    crop = cv2.imread(
        str(crop_path),
        cv2.IMREAD_COLOR
    )

    if crop is None:

        print(
            f"  ❌ Cannot read crop:"
            f" {crop_path}"
        )

        return False

    # ========================================================
    # LOAD ALREADY-CROPPED MASK
    #
    # IMPORTANT:
    #
    # This mask is ALREADY aligned with the crop.
    #
    # DO NOT use crop_bbox here.
    # ========================================================

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        print(
            f"  ❌ Cannot read mask:"
            f" {mask_path}"
        )

        return False

    crop_h, crop_w = crop.shape[:2]

    mask_h, mask_w = mask.shape[:2]

    print(
        f"  Crop resolution: "
        f"{crop_w} × {crop_h}"
    )

    print(
        f"  Mask resolution: "
        f"{mask_w} × {mask_h}"
    )

    # ========================================================
    # CRITICAL ALIGNMENT CHECK
    # ========================================================

    if (
        crop_h != mask_h
        or
        crop_w != mask_w
    ):

        print(
            "  ❌ CROP/MASK SIZE MISMATCH"
        )

        print(
            f"     Crop: "
            f"{crop_w} × {crop_h}"
        )

        print(
            f"     Mask: "
            f"{mask_w} × {mask_h}"
        )

        print(
            "     Model 1 output is invalid."
        )

        return False

    # ========================================================
    # FORCE BINARY MASK
    # ========================================================

    binary_mask = np.where(
        mask > 127,
        255,
        0
    ).astype(np.uint8)

    # ========================================================
    # CHECK MASK
    # ========================================================

    cable_pixels = np.count_nonzero(
        binary_mask
    )

    total_pixels = binary_mask.size

    if cable_pixels == 0:

        print(
            "  ❌ Mask contains ZERO cable pixels."
        )

        return False

    coverage = (
        100.0
        *
        cable_pixels
        /
        total_pixels
    )

    print(
        f"  Cable mask coverage: "
        f"{coverage:.2f}%"
    )

    # ========================================================
    # CREATE WHITE BACKGROUND
    # ========================================================

    white_background = np.full(
        crop.shape,
        255,
        dtype=np.uint8
    )

    # ========================================================
    # KEEP ONLY RGB CABLE PIXELS
    #
    # mask = 255
    #       → original RGB pixel
    #
    # mask = 0
    #       → white
    # ========================================================

    model2_input = np.where(
        binary_mask[:, :, None] > 0,
        crop,
        white_background
    )

    model2_input = model2_input.astype(
        np.uint8
    )

    # ========================================================
    # VERIFY OUTPUT
    # ========================================================

    if model2_input.shape != crop.shape:

        print(
            "  ❌ Model 2 image shape error"
        )

        return False

    # ========================================================
    # RESIZE RGB IMAGE TO 640×640
    # ========================================================

    model2_input_640 = letterbox_white(
        model2_input,
        IMG_SIZE,
        is_mask=False
    )

    # ========================================================
    # RESIZE MASK TO 640×640
    # ========================================================

    mask_640 = letterbox_white(
        binary_mask,
        IMG_SIZE,
        is_mask=True
    )

    # Force binary again after resizing
    mask_640 = np.where(
        mask_640 > 127,
        255,
        0
    ).astype(np.uint8)

    # ========================================================
    # FILENAMES
    # ========================================================

    stem = crop_path.stem

    image_output = (
        MODEL2_IMAGES
        /
        f"{stem}.png"
    )

    mask_output = (
        MODEL2_MASKS
        /
        f"{stem}_mask.png"
    )

    preview_output = (
        MODEL2_PREVIEWS
        /
        f"{stem}_preview.jpg"
    )

    # ========================================================
    # SAVE MODEL 2 RGB + WHITE IMAGE
    #
    # PNG avoids JPEG compression during dataset creation.
    # ========================================================

    saved_image = cv2.imwrite(
        str(image_output),
        model2_input_640
    )

    if not saved_image:

        print(
            "  ❌ Failed to save Model 2 image"
        )

        return False

    # ========================================================
    # SAVE MASK
    # ========================================================

    saved_mask = cv2.imwrite(
        str(mask_output),
        mask_640
    )

    if not saved_mask:

        print(
            "  ❌ Failed to save Model 2 mask"
        )

        return False

    # ========================================================
    # SAVE PREVIEW
    #
    # This is EXACTLY the image Model 2 receives.
    # ========================================================

    saved_preview = cv2.imwrite(
        str(preview_output),
        model2_input_640,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not saved_preview:

        print(
            "  ⚠️ Failed to save preview"
        )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        f"  ✅ Cable {cable_id}"
    )

    print(
        f"     Original crop:"
        f" {crop_w} × {crop_h}"
    )

    print(
        f"     Model 2 input:"
        f" {IMG_SIZE} × {IMG_SIZE}"
    )

    print(
        f"     RGB + white:"
        f" {image_output.name}"
    )

    print(
        f"     Mask:"
        f" {mask_output.name}"
    )

    return True


# ============================================================
# LOAD MODEL 1 JSON
# ============================================================

print()

print(
    "=" * 70
)

print(
    "MODEL 2 DATASET PREPROCESSING"
)

print(
    "=" * 70
)

print(
    f"\nModel 1 JSON:"
    f"\n{RESULTS_JSON}"
)

if not RESULTS_JSON.exists():

    raise FileNotFoundError(
        f"\nModel 1 JSON not found:\n"
        f"{RESULTS_JSON}"
    )

with open(
    RESULTS_JSON,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)

print(
    f"\nImages in JSON: "
    f"{len(results)}"
)


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

total_cables = 0

successful = 0

failed = 0


for image_result in results:

    image_name = image_result[
        "image"
    ]

    status = image_result.get(
        "status",
        ""
    )

    # ========================================================
    # ONLY PROCESS SUCCESSFUL MODEL 1 IMAGES
    # ========================================================

    if status != "SUCCESS":

        print(
            f"\n⚠️ Skipping:"
            f" {image_name}"
            f" | status={status}"
        )

        continue

    cables = image_result.get(
        "cables",
        []
    )

    print()

    print(
        "-" * 70
    )

    print(
        f"Processing: "
        f"{image_name}"
    )

    print(
        f"Cables: "
        f"{len(cables)}"
    )

    print(
        "-" * 70
    )

    for cable in cables:

        total_cables += 1

        result = process_cable(
            cable,
            image_name
        )

        if result:

            successful += 1

        else:

            failed += 1


# ============================================================
# FINAL REPORT
# ============================================================

print()

print(
    "=" * 70
)

print(
    "MODEL 2 PREPROCESSING COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Total cables : "
    f"{total_cables}"
)

print(
    f"Successful   : "
    f"{successful}"
)

print(
    f"Failed       : "
    f"{failed}"
)

print()

print(
    "MODEL 2 RGB + WHITE IMAGES:"
)

print(
    MODEL2_IMAGES
)

print()

print(
    "MODEL 2 MASKS:"
)

print(
    MODEL2_MASKS
)

print()

print(
    "PREVIEWS:"
)

print(
    MODEL2_PREVIEWS
)

print(
    "=" * 70
)
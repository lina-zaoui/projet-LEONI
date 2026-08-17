import cv2
import json

from src.model1_segmentation import CableSegmenter

from src.cable_crop import (
    crop_cable,
    create_mask_overlay
)

from src.visualization import create_visualization

from config import (
    MODEL_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    MASK_DIR,
    CROP_DIR,
    VISUALIZATION_DIR,
    RESULTS_DIR,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CABLE_CLASS_ID,
    CROP_PADDING,
    EXPECTED_CABLES
)


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    segmenter,
    image_path
):
    """
    Process one complete original camera image.

    Pipeline:

        Original 5472x2188 image
                    |
                    v
               YOLO Model 1
                    |
              bbox + polygon
                    |
          +---------+---------+
          |                   |
          v                   v
      bbox crop          mask crop
      from original      from polygon
          |                   |
          v                   v
    High-resolution      Mask overlay
        crop             visualization
          |
          v
       Model 2
    """

    print("\n" + "-" * 70)

    print(
        f"Processing: {image_path.name}"
    )

    # ========================================================
    # LOAD ORIGINAL IMAGE
    # ========================================================

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "❌ Could not read image"
        )

        return {
            "image": image_path.name,
            "status": "IMAGE_READ_FAILED",
            "cables_detected": 0,
            "cables": []
        }

    print(
        f"✅ Original image loaded: "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    # ========================================================
    # MODEL 1
    # ========================================================

    print(
        "\nRunning YOLO11 segmentation..."
    )

    detections = segmenter.predict(
        image
    )

    # ========================================================
    # CHECK DETECTIONS
    # ========================================================

    if not detections:

        print(
            "❌ No complete cable detected"
        )

        return {
            "image": image_path.name,
            "status": "NO_CABLE",
            "cables_detected": 0,
            "expected_cables": EXPECTED_CABLES,
            "cables": []
        }

    print(
        f"✅ {len(detections)} cable(s) detected"
    )

    # ========================================================
    # CHECK EXPECTED NUMBER
    # ========================================================

    if len(detections) != EXPECTED_CABLES:

        print(
            f"⚠️ Expected "
            f"{EXPECTED_CABLES} cables, "
            f"but detected "
            f"{len(detections)}"
        )

    cable_results = []

    # ========================================================
    # PROCESS EVERY CABLE
    # ========================================================

    for cable_index, detection in enumerate(
        detections,
        start=1
    ):

        confidence = (
            detection["confidence"]
        )

        polygon = (
            detection["polygon"]
        )

        bbox = (
            detection["bbox"]
        )

        print(
            f"\n  Cable {cable_index}"
        )

        print(
            f"  Confidence: "
            f"{confidence:.3f}"
        )

        print(
            f"  Bounding box: "
            f"{bbox}"
        )

        # ====================================================
        # CROP ORIGINAL IMAGE + MASK
        # ====================================================

        crop, mask, padded_bbox = crop_cable(
            image=image,
            polygon=polygon,
            bbox=bbox,
            padding=CROP_PADDING
        )

        if crop.size == 0:

            print(
                "  ❌ Empty crop"
            )

            continue

        print(
            f"  Crop size: "
            f"{crop.shape[1]} x "
            f"{crop.shape[0]}"
        )

        print(
            f"  Mask size: "
            f"{mask.shape[1]} x "
            f"{mask.shape[0]}"
        )

        # ====================================================
        # FILE NAMES
        # ====================================================

        base_name = (
            f"{image_path.stem}"
            f"_cable_{cable_index}"
        )

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask_path = (
            MASK_DIR /
            f"{base_name}_mask.png"
        )

        # ----------------------------------------------------
        # HIGH-RESOLUTION CROP
        #
        # PNG is used instead of JPG to avoid JPEG compression.
        # ----------------------------------------------------

        crop_path = (
            CROP_DIR /
            f"{base_name}.png"
        )

        # ----------------------------------------------------
        # MASK OVERLAY
        # ----------------------------------------------------

        overlay_path = (
            VISUALIZATION_DIR /
            f"{base_name}_mask_overlay.jpg"
        )

        # ====================================================
        # SAVE MASK
        # ====================================================

        mask_saved = cv2.imwrite(
            str(mask_path),
            mask
        )

        if mask_saved:

            print(
                f"  ✅ Mask: "
                f"{mask_path}"
            )

        else:

            print(
                f"  ❌ Failed to save mask"
            )

        # ====================================================
        # SAVE HIGH-RESOLUTION CROP
        # ====================================================

        crop_saved = cv2.imwrite(
            str(crop_path),
            crop
        )

        if crop_saved:

            print(
                f"  ✅ High-resolution crop: "
                f"{crop_path}"
            )

        else:

            print(
                f"  ❌ Failed to save crop"
            )

            continue

        # ====================================================
        # CREATE MASK OVERLAY
        #
        # This is only for visualization.
        #
        # It does NOT modify the crop.
        # It does NOT modify the mask.
        # ====================================================

        overlay = create_mask_overlay(
            crop,
            mask
        )

        overlay_saved = cv2.imwrite(
            str(overlay_path),
            overlay
        )

        if overlay_saved:

            print(
                f"  ✅ Mask overlay: "
                f"{overlay_path}"
            )

        else:

            print(
                f"  ❌ Failed to save mask overlay"
            )

        # ====================================================
        # STORE MODEL 1 RESULT
        # ====================================================

        cable_results.append({

            "cable_id": cable_index,

            "class_id": CABLE_CLASS_ID,

            "class_name": "cable",

            "confidence": round(
                confidence,
                4
            ),

            # Original YOLO bbox
            "bbox": [
                int(v)
                for v in bbox
            ],

            # Bbox actually used for cropping
            "crop_bbox": [
                int(v)
                for v in padded_bbox
            ],

            # High-resolution crop
            "crop": str(
                crop_path
            ),

            # Cropped segmentation mask
            "mask": str(
                mask_path
            ),

            # Visualization for checking mask quality
            "mask_overlay": str(
                overlay_path
            )
        })

    # ========================================================
    # FULL IMAGE VISUALIZATION
    # ========================================================

    visualization = create_visualization(
        image,
        detections
    )

    visualization_path = (
        VISUALIZATION_DIR /
        f"{image_path.stem}_result.jpg"
    )

    visualization_saved = cv2.imwrite(
        str(visualization_path),
        visualization
    )

    if visualization_saved:

        print(
            f"\n✅ Full image visualization: "
            f"{visualization_path}"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    successful_crops = len(
        cable_results
    )

    if successful_crops == EXPECTED_CABLES:

        status = "SUCCESS"

    else:

        status = "INCOMPLETE"

    return {

        "image": image_path.name,

        "status": status,

        "expected_cables": EXPECTED_CABLES,

        "cables_detected": len(
            detections
        ),

        "successful_crops": (
            successful_crops
        ),

        "visualization": str(
            visualization_path
        ),

        "cables": cable_results
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================

    MASK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CROP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # DISPLAY CONFIGURATION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL 1 - YOLO11 CABLE SEGMENTATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Model      : {MODEL_PATH}"
    )

    print(
        f"Input      : {INPUT_DIR}"
    )

    print(
        f"Output     : {OUTPUT_DIR}"
    )

    print(
        f"Image size : {IMAGE_SIZE}"
    )

    print(
        f"Confidence : {CONFIDENCE_THRESHOLD}"
    )

    print(
        f"IoU        : {IOU_THRESHOLD}"
    )

    print(
        f"Crop pad   : {CROP_PADDING}"
    )

    print(
        f"Expected   : "
        f"{EXPECTED_CABLES} cables"
    )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        print(
            "\n❌ MODEL NOT FOUND"
        )

        print(
            MODEL_PATH
        )

        return

    print(
        "\n✅ Model found"
    )

    # ========================================================
    # CHECK INPUT DIRECTORY
    # ========================================================

    if not INPUT_DIR.exists():

        print(
            "\n❌ INPUT DIRECTORY NOT FOUND"
        )

        print(
            INPUT_DIR
        )

        return

    print(
        "✅ Input directory found"
    )

    # ========================================================
    # FIND IMAGES
    # ========================================================

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    image_paths = [

        path

        for path in INPUT_DIR.rglob("*")

        if (
            path.is_file()
            and
            path.suffix.lower()
            in valid_extensions
        )
    ]

    print(
        f"\n📁 Images found: "
        f"{len(image_paths)}"
    )

    if not image_paths:

        print(
            "\n❌ No images found"
        )

        return

    for path in image_paths[:10]:

        print(
            f"  → {path}"
        )

    if len(image_paths) > 10:

        print(
            f"  ... "
            f"{len(image_paths) - 10} more"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading YOLO11 model..."
    )

    segmenter = CableSegmenter(

        model_path=MODEL_PATH,

        image_size=IMAGE_SIZE,

        confidence=CONFIDENCE_THRESHOLD,

        iou=IOU_THRESHOLD,

        cable_class_id=CABLE_CLASS_ID
    )

    print(
        "✅ YOLO11 model loaded"
    )

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    all_results = []

    total_cables = 0

    complete_images = 0

    incomplete_images = 0

    no_cable_images = 0

    # ========================================================
    # IMAGE LOOP
    # ========================================================

    for image_path in image_paths:

        result = process_image(
            segmenter,
            image_path
        )

        all_results.append(
            result
        )

        total_cables += result.get(
            "cables_detected",
            0
        )

        if result["status"] == "SUCCESS":

            complete_images += 1

        elif result["status"] == "INCOMPLETE":

            incomplete_images += 1

        elif result["status"] == "NO_CABLE":

            no_cable_images += 1

    # ========================================================
    # SAVE JSON
    # ========================================================

    results_path = (
        RESULTS_DIR /
        "model1_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL 1 FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        f"Images processed : "
        f"{len(image_paths)}"
    )

    print(
        f"Complete images  : "
        f"{complete_images}"
    )

    print(
        f"Incomplete images: "
        f"{incomplete_images}"
    )

    print(
        f"No cable images  : "
        f"{no_cable_images}"
    )

    print(
        f"Total cables     : "
        f"{total_cables}"
    )

    print(
        "\nHigh-resolution crops:"
    )

    print(
        CROP_DIR
    )

    print(
        "\nMasks:"
    )

    print(
        MASK_DIR
    )

    print(
        "\nMask overlays:"
    )

    print(
        VISUALIZATION_DIR
    )

    print(
        "\nJSON:"
    )

    print(
        results_path
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
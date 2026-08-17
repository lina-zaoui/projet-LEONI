from pathlib import Path
import numpy as np
from ultralytics import YOLO


class CableSegmenter:
    """
    YOLO11 segmentation model for detecting complete cables.

    The original camera image is passed directly to YOLO.
    YOLO internally resizes/letterboxes the image for inference,
    but the returned bbox/polygon coordinates are mapped back
    to the original image resolution.

    This means the crop is always extracted from the
    ORIGINAL HIGH-RESOLUTION IMAGE.
    """

    def __init__(
        self,
        model_path,
        image_size=1024,
        confidence=0.20,
        iou=0.50,
        cable_class_id=0
    ):

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"\nModel not found:\n"
                f"{model_path}\n\n"
                f"Expected location:\n"
                f"{model_path.resolve()}"
            )

        # Load YOLO model
        self.model = YOLO(str(model_path))

        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.cable_class_id = cable_class_id

        print(f"Loaded model: {model_path}")
        print(f"Model classes: {self.model.names}")
        print(f"Inference image size: {self.image_size}")

    def predict(self, image):
        """
        Run YOLO11 segmentation on one ORIGINAL image.

        Important:
        The input image is NOT manually resized before inference.

        YOLO performs its own internal preprocessing and returns
        coordinates mapped back to the original image dimensions.

        Returns:
            List of cable detections.

        Each detection:

            {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],
                "polygon": np.ndarray
            }
        """

        if image is None:
            return []

        original_height, original_width = image.shape[:2]

        print(
            f"  YOLO input: "
            f"{original_width} × {original_height}"
        )

        # ====================================================
        # YOLO INFERENCE
        # ====================================================

        results = self.model.predict(
            source=image,

            # Keep this at 1024.
            # Do NOT resize the original image yourself.
            imgsz=self.image_size,

            conf=self.confidence,
            iou=self.iou,

            # Important for keeping returned masks/bboxes
            # correctly associated with the original image.
            retina_masks=True,

            verbose=False
        )

        if not results:
            print("YOLO detections: 0")
            return []

        result = results[0]

        # ====================================================
        # CHECK BOXES
        # ====================================================

        if result.boxes is None:
            print("YOLO detections: 0")
            return []

        if len(result.boxes) == 0:
            print("YOLO detections: 0")
            return []

        # ====================================================
        # CHECK MASKS
        # ====================================================

        if result.masks is None:
            print(
                "YOLO detected objects, "
                "but no segmentation masks were returned."
            )
            return []

        # ====================================================
        # GET BOUNDING BOXES
        # ====================================================

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        class_ids = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        # ====================================================
        # GET POLYGONS
        # ====================================================

        # YOLO's masks.xy are already expressed in the
        # original image coordinate system.
        polygons = result.masks.xy

        detections = []

        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        for i in range(len(boxes)):

            class_id = int(class_ids[i])

            # Only keep cable class
            if class_id != self.cable_class_id:
                continue

            polygon = polygons[i]

            if polygon is None or len(polygon) < 3:

                print(
                    f"Skipping detection {i}: "
                    f"invalid polygon"
                )

                continue

            x1, y1, x2, y2 = boxes[i]

            # =================================================
            # CLAMP BBOX TO ORIGINAL IMAGE
            # =================================================

            x1 = max(
                0,
                min(x1, original_width - 1)
            )

            y1 = max(
                0,
                min(y1, original_height - 1)
            )

            x2 = max(
                1,
                min(x2, original_width)
            )

            y2 = max(
                1,
                min(y2, original_height)
            )

            # =================================================
            # CLAMP POLYGON TO ORIGINAL IMAGE
            # =================================================

            polygon = np.asarray(
                polygon,
                dtype=np.float32
            ).copy()

            polygon[:, 0] = np.clip(
                polygon[:, 0],
                0,
                original_width - 1
            )

            polygon[:, 1] = np.clip(
                polygon[:, 1],
                0,
                original_height - 1
            )

            detection = {

                "class_id": class_id,

                "class_name": self.model.names[
                    class_id
                ],

                "confidence": float(
                    confidences[i]
                ),

                "bbox": [
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2))
                ],

                "polygon": polygon
            }

            detections.append(
                detection
            )

        # ====================================================
        # SORT CABLES
        # ====================================================

        def sort_key(detection):

            x1, y1, x2, y2 = detection["bbox"]

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            return (
                center_y,
                center_x
            )

        detections.sort(
            key=sort_key
        )

        print(
            f"YOLO detected "
            f"{len(detections)} cable(s)"
        )

        # ====================================================
        # DEBUG INFORMATION
        # ====================================================

        for i, detection in enumerate(
            detections,
            start=1
        ):

            print(
                f"  Cable {i}: "
                f"confidence="
                f"{detection['confidence']:.3f}, "
                f"bbox="
                f"{detection['bbox']}"
            )

        return detections
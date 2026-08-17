import cv2
import numpy as np


def create_visualization(
    image,
    detections
):
    """
    Draw all cable masks and bounding boxes on the original image.

    Each cable receives its own visualization label.
    """

    output = image.copy()

    # A small set of colors for up to four cables.
    # These are only for visualization.
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 165, 255),
        (255, 0, 255)
    ]

    for index, detection in enumerate(detections):

        color = colors[
            index % len(colors)
        ]

        polygon = detection["polygon"]

        confidence = detection["confidence"]

        bbox = detection["bbox"]

        # ------------------------------------------------
        # CREATE MASK
        # ------------------------------------------------

        mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8
        )

        points = np.round(
            polygon
        ).astype(np.int32)

        cv2.fillPoly(
            mask,
            [points],
            255
        )

        # ------------------------------------------------
        # MASK OVERLAY
        # ------------------------------------------------

        overlay = output.copy()

        overlay[mask > 0] = (
            0.45 * overlay[mask > 0]
            + 0.55 * np.array(
                color,
                dtype=np.float32
            )
        ).astype(np.uint8)

        output = overlay

        # ------------------------------------------------
        # POLYGON OUTLINE
        # ------------------------------------------------

        cv2.polylines(
            output,
            [points],
            isClosed=True,
            color=color,
            thickness=2
        )

        # ------------------------------------------------
        # BOUNDING BOX
        # ------------------------------------------------

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ------------------------------------------------
        # LABEL
        # ------------------------------------------------

        label = (
            f"Cable {index + 1} "
            f"{confidence:.2f}"
        )

        text_y = max(
            25,
            y1 - 8
        )

        cv2.putText(
            output,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA
        )

    return output
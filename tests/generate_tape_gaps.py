import cv2
import numpy as np

IMAGE_PATH = r"C:\Users\Administrator\Desktop\stage-LEONI\projet-LEONI\data\input_model2\normal\1_png_cable_1_png.rf.2911b41793d6781957f083e5795bb806.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image could not be loaded.")
    exit()

print("Image shape:", image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Gray minimum:", gray.min())
print("Gray maximum:", gray.max())
print("Gray mean:", gray.mean())

# Try several thresholds
for threshold in [50, 80, 100, 120, 150, 180, 200, 220]:

    _, mask = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    pixels = np.sum(mask > 0)
    percentage = pixels / mask.size * 100

    print(
        f"Threshold {threshold}: "
        f"{pixels} dark pixels "
        f"({percentage:.2f}%)"
    )

cv2.imwrite(
    "diagnostic_gray.jpg",
    gray
)

cv2.imwrite(
    "diagnostic_mask.jpg",
    mask
)

print("\nSaved:")
print("diagnostic_gray.jpg")
print("diagnostic_mask.jpg")
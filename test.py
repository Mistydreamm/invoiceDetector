# =============================================================================
# Test File for Paddle OCR 02/07/2025
# =============================================================================

from paddleocr import PaddleOCR
import cv2
from matplotlib import pyplot as plt

ocr = PaddleOCR(use_angle_cls=True, lang="fr")
image_path = 'imageTest.png'
img = cv2.imread(image_path)

plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.show()

result = ocr.predict(img)

# Print all recognized texts
for text in result[0]['rec_texts']:
    print(text)

# import cv2
# import os
# from main import InpaintCleaner, Typesetter

# # Create test image
# os.makedirs("./output_tiles", exist_ok=True)
# test_img = cv2.imread("./output_tiles/02-optimized_009.png")

# # Mock data from OCR
# mock_data = [
#     {
#         "id": 1,
#         "box": {"x": 90, "y": 120, "w": 250, "h": 50},
#         "original_text": "MISSION #1",
#         "translated_text": "ภารกิจที่ 1",
#     }
# ]

# # Processing
# cleaner = InpaintCleaner()
# typesetter = Typesetter(font_path="./THSarabunNew Bold.ttf")

# result = test_img.copy()
# for item in mock_data:
#     result = cleaner.clean(result, item["box"])
#     result = typesetter.draw_text(result, item["translated_text"], item["box"])

# cv2.imwrite("result.png", result)

from main import ContentAwareSlicer


slicer = ContentAwareSlicer(
    target_height=1200,  # Slice every ~1200px
    search_window=400,  # Allow ±400px flexibility
)

slicer.process(image_path="./image/07-optimized.webp", output_dir="./output_tiles/07")
# slicer.process(image_path="./image/06.jpg", output_dir="./output_tiles/06")

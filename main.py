import cv2
import numpy as np
import os
import easyocr
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
import json

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_words
from pythainlp.util import dict_trie


# ==========================================
# 1. Image Processing & Cleaning Strategies
# ==========================================
class CleanStrategy:
    def clean(self, image, bbox):
        pass


class SolidFillCleaner(CleanStrategy):
    def clean(self, image, bbox):
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)
        return image


class InpaintCleaner(CleanStrategy):
    def clean(self, image, bbox):
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


class ContentAwareSlicer:
    def __init__(self, target_height=1500, search_window=300):
        self.target_height = target_height
        self.search_window = search_window

    def _find_safe_cut_y(self, edges_image, start_y, expected_y):
        height, _ = edges_image.shape
        min_y = max(start_y + 100, expected_y - self.search_window)
        max_y = min(height - 10, expected_y + self.search_window)
        if min_y >= max_y:
            return expected_y
        search_area = edges_image[min_y:max_y, :]
        row_sums = np.sum(search_area, axis=1)
        return min_y + np.argmin(row_sums)

    def process(self, image_path, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image")
        height, width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        current_y = 0
        slice_count = 1
        while current_y < height:
            expected_y = current_y + self.target_height
            if expected_y >= height:
                safe_cut_y = height
            else:
                safe_cut_y = self._find_safe_cut_y(edges, current_y, expected_y)
            tile = img[current_y:safe_cut_y, :]
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_file = os.path.join(output_dir, f"{base_name}_{slice_count:03d}.png")
            cv2.imwrite(out_file, tile)
            current_y = safe_cut_y
            slice_count += 1


# ==========================================
# 2. OCR & Translation Engine (Phase 1: Extractor)
# ==========================================
class MangaOcrEngine:
    def __init__(self):
        print("Loading OCR Engine...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        print("Loading Translation Engine...")
        self.translator = GoogleTranslator(source="en", target="th")

    def extract_to_json_format(self, image_path):
        print(f"Scanning image for paragraphs (Custom Clustering)...")
        raw_results = self.reader.readtext(image_path)

        raw_boxes = []
        for bbox_coords, text, prob in raw_results:
            xs = [point[0] for point in bbox_coords]
            ys = [point[1] for point in bbox_coords]
            raw_boxes.append(
                {
                    "x": int(min(xs)),
                    "y": int(min(ys)),
                    "w": int(max(xs) - min(xs)),
                    "h": int(max(ys) - min(ys)),
                    "text": text,
                }
            )

        raw_boxes = sorted(raw_boxes, key=lambda b: b["y"])
        bubbles = []

        for box in raw_boxes:
            if not bubbles:
                bubbles.append(box)
                continue

            last = bubbles[-1]
            vertical_gap = box["y"] - (last["y"] + last["h"])
            horizontal_overlap = max(
                0,
                min(last["x"] + last["w"], box["x"] + box["w"])
                - max(last["x"], box["x"]),
            )

            if vertical_gap < 80 and horizontal_overlap > -50:
                new_x = min(last["x"], box["x"])
                new_y = min(last["y"], box["y"])
                new_w = max(last["x"] + last["w"], box["x"] + box["w"]) - new_x
                new_h = max(last["y"] + last["h"], box["y"] + box["h"]) - new_y

                last["x"], last["y"], last["w"], last["h"] = new_x, new_y, new_w, new_h
                last["text"] += " " + box["text"]
            else:
                bubbles.append(box)

        extracted_data = []
        for idx, b in enumerate(bubbles):
            try:
                translated_draft = self.translator.translate(b["text"])
            except Exception as e:
                translated_draft = b["text"]

            extracted_data.append(
                {
                    "id": idx + 1,
                    "box": {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]},
                    "original_text": b["text"],
                    "translated_text": translated_draft,
                }
            )
            print(f"-> Merged Bubble {idx + 1}: '{b['text']}'")

        return extracted_data


# ==========================================
# 3. Typesetter Engine (Smart Tone Positioning)
# ==========================================
class Typesetter:
    def __init__(self, font_path=None, base_font_size=28, min_font_size=14):
        self.font_path = font_path
        self.base_font_size = base_font_size
        self.min_font_size = min_font_size

        custom_words = set(thai_words())
        custom_words.update(["บอกว่า", "ดีแล้ว", "เกินจริง"])
        if "บอ" in custom_words:
            custom_words.remove("บอ")
        self.custom_dict = dict_trie(custom_words)

    def _draw_thai_text_robust(self, draw, text, x, y, font, fill=(0, 0, 0)):
        upper_vowels = ["\u0e31", "\u0e34", "\u0e35", "\u0e36", "\u0e37", "\u0e4d"]
        tone_marks = ["\u0e48", "\u0e49", "\u0e4a", "\u0e4b", "\u0e4c"]

        base_text = ""
        tones_to_draw = []

        for char in text:
            if char in tone_marks:
                if len(base_text) > 0 and base_text[-1] in upper_vowels:
                    tones_to_draw.append((char, len(base_text), "with_vowel"))
                else:
                    tones_to_draw.append((char, len(base_text), "no_vowel"))
            else:
                base_text += char

        draw.text((x, y), base_text, font=font, fill=fill)

        shift_up = int(font.size * 0.12)
        shift_down = int(font.size * 0.15)

        for tone_char, index, condition in tones_to_draw:
            prefix = base_text[:index]
            offset_x = draw.textlength(prefix, font=font)

            if condition == "with_vowel":
                final_y = y - shift_up
            else:
                final_y = y + shift_down

            draw.text((x + offset_x, final_y), tone_char, font=font, fill=fill)

    def _wrap_text(self, draw, text, font, max_width):
        words = word_tokenize(text, engine="newmm", custom_dict=self.custom_dict)
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word if current_line else word
            line_w = draw.textbbox((0, 0), test_line, font=font)[2]

            if line_w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def draw_text(self, cv2_image, text, bbox):
        color_coverted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        draw = ImageDraw.Draw(pil_image)
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        if not self.font_path or not os.path.exists(self.font_path):
            return cv2_image

        current_size = self.base_font_size
        font = ImageFont.truetype(self.font_path, size=current_size)
        lines = []

        while current_size >= self.min_font_size:
            font = ImageFont.truetype(self.font_path, size=current_size)
            lines = self._wrap_text(draw, text, font, w * 0.9)
            line_height = draw.textbbox((0, 0), "ไ", font=font)[3] + 2
            total_height = len(lines) * line_height
            if total_height <= h * 0.9:
                break
            current_size -= 2

        line_height = draw.textbbox((0, 0), "ไ", font=font)[3] + 2
        total_height = len(lines) * line_height
        current_y = y + (h - total_height) / 2

        for line in lines:
            line_w = draw.textbbox((0, 0), line, font=font)[2]
            line_x = x + (w - line_w) / 2

            self._draw_thai_text_robust(
                draw, line, line_x, current_y, font, fill=(0, 0, 0)
            )

            current_y += line_height

        numpy_image = np.array(pil_image)
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)


# ==========================================
# 4. Main Workflow (Human-in-the-Loop)
# ==========================================
if __name__ == "__main__":
    target_image_path = "./output_tiles/02-optimized_009.png"
    json_path = "translations.json"

    if not os.path.exists(target_image_path):
        print(f"✗ ERROR: File not found: {target_image_path}")
        exit(1)

    # ---------------------------------------------------------
    # Phase 1: AI Extraction & Drafting (Creates JSON)
    # ---------------------------------------------------------
    if not os.path.exists(json_path):
        print("\n--- Phase 1: AI Extraction & Drafting ---")
        ocr_engine = MangaOcrEngine()
        extracted_data = ocr_engine.extract_to_json_format(target_image_path)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)

        print(f"\n✅ SUCCESS: Created '{json_path}' successfully!")
        print(
            "⏸️ SYSTEM PAUSED: Please review and manually edit the translations in translations.json"
        )
        print("   (Once done, save the file and re-run this script)")
        exit(0)

    # ---------------------------------------------------------
    # Phase 2: Human-in-the-Loop Injection (Draws Text)
    # ---------------------------------------------------------
    else:
        print("\n--- Phase 2: Human-in-the-Loop Injection ---")

        with open(json_path, "r", encoding="utf-8") as f:
            edited_data = json.load(f)

        original_img = cv2.imread(target_image_path)
        current_img = original_img.copy()

        cleaner = InpaintCleaner()
        typesetter = Typesetter(font_path="./THSarabunNew Bold.ttf", base_font_size=28)

        for item in edited_data:
            print(f"-> Injecting text: '{item['translated_text']}'")
            current_img = cleaner.clean(current_img, item["box"])
            current_img = typesetter.draw_text(
                current_img, item["translated_text"], item["box"]
            )

        output_file = "result_final_pipeline.png"
        cv2.imwrite(output_file, current_img)
        print(
            f"\n🎉 DONE! Pipeline completed successfully. Check the output at: {output_file}"
        )

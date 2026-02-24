# 💬 Manga Auto-Translator & Smart Thai Typesetter

An end-to-end Python pipeline for translating manga/comics from English to Thai. This project features a robust **Human-in-the-Loop (HITL)** architecture and a custom-built **Smart Thai Typesetting Algorithm** that completely resolves the notorious Thai vowel/tone overlap issues on Windows without relying on OS-level C-libraries (like `libraqm`).

---

## 🌟 Key Features

- **✂️ Content-Aware Image Slicing**  
  Automatically detects blank spaces to slice long webtoon strips into manageable pages without cutting through text or speech bubbles.

- **🧠 Contextual OCR Clustering**  
  Upgrades default EasyOCR line-by-line reading into a custom Paragraph-level clustering algorithm, preventing context loss before translation.

- **⏸️ Human-in-the-Loop (HITL) Workflow**  
  Decouples the extraction and rendering processes. The AI generates a `translations.json` draft, pausing the pipeline to allow human editors to refine translations for maximum naturalness before final rendering.

- **🪄 Seamless Inpainting**  
  Uses OpenCV's Telea algorithm to cleanly erase original text from speech bubbles.

- **🇹🇭 Smart Thai Typesetting (Strip & Shift Algorithm)**
  - Custom AI Dictionary-based word wrapping (`pythainlp`) to prevent awkward line breaks
  - Intelligently calculates coordinates to shift Thai tone marks (วรรณยุกต์) based on vowel presence
  - Achieves pixel-perfect typography on any OS without external C-libraries

---

## 🛠️ Technology Stack

| Component               | Library              | Version           |
| ----------------------- | -------------------- | ----------------- |
| **Language**            | Python               | 3.11+             |
| **Computer Vision**     | OpenCV               | `opencv-python`   |
| **OCR**                 | EasyOCR              | Latest            |
| **Translation**         | Google Translate API | `deep-translator` |
| **NLP (Thai)**          | PyThaiNLP            | Latest            |
| **Image Processing**    | Pillow               | `PIL`             |
| **Numerical Computing** | NumPy                | Latest            |

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11 or higher
- A Thai font file (e.g., `THSarabunNew Bold.ttf` or `tahoma.ttf`)

### 2. Setup Environment
```bash
# Install uv (fast Python package manager)
pip install uv

# Create virtual environment with Python 3.11
uv python install 3.11
uv venv --python 3.11

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv sync
```

### 3. Font Setup
Place a Thai font file in your project root directory and update the `font_path` variable in `main.py`:
```python
Typesetter(font_path="THSarabunNew Bold.ttf")
```

---

## 🕹️ Workflow: 2-Phase Pipeline

### Phase 1️⃣: AI Extraction & Drafting
Run the script for the first time. The OCR will:
1. Scan the image
2. Cluster paragraphs using custom logic
3. Generate draft translations
4. Save results to `translations.json`
5. **Pause and wait for human review**

```bash
uv run main.py
```

### Phase 1.5️⃣: Human Review (Crucial!)
Open `translations.json` in your code editor:
```json
{
  "id": 1,
  "original_text": "Hello world",
  "translated_text": "สวัสดีชาวโลก"  ← Edit this field
}
```
Fix any robotic translations, then save the file.

### Phase 2️⃣: Injection & Typesetting
Run the script again. It will:
1. Detect the existing `translations.json`
2. Bypass the OCR phase (faster!)
3. Erase original text with inpainting
4. Render beautifully typeset Thai text

```bash
uv run main.py
# Output: result_final_pipeline.png
```

---

## 📁 Project Structure

```
converter-ocr/
├── main.py                      # Main pipeline script
├── pyproject.toml               # Project config & dependencies
├── .gitignore                   # Git ignore rules
├── README.md                    # Quick reference (this file)
├── docs/
│   ├── README.md                # English documentation
│   ├── README_TH.md             # Thai documentation (ภาษาไทย)
│   └── ARCHITECTURE.md           # Technical deep-dive (optional)
├── image/
│   └── 02-optimized.webp        # Sample manga/webtoon page
├── output_tiles/                # Generated image tiles
└── translations.json            # HITL checkpoint (user edits here)
```

---

## 🧩 Customization & Extensibility

This project follows **SOLID** principles (Single Responsibility, Strategy Pattern) for easy customization:

### 🔄 Swap the Translation Engine
Replace `GoogleTranslator` in `MangaOcrEngine` with:
- **OpenAI GPT-4o** for better context understanding
- **Claude 3** for nuanced translations
- **DeepL** for higher quality

**Example:**
```python
class MangaOcrEngine:
    def __init__(self):
        self.translator = OpenAITranslator()  # Swap here
```

### 🎯 Fine-tune OCR Clustering
Adjust the `vertical_gap` threshold if bubbles are being merged too aggressively:
```python
# In MangaOcrEngine.extract_to_json_format()
if vertical_gap < 80:  # ← Increase to 120 for tighter clustering
    # Merge bubbles
```

### 📚 Enhance NLP Dictionary
Add custom words for consistent character names:
```python
# In Typesetter.__init__()
custom_words.update(["ชื่ออักษร", "ชื่ออื่น"])  # Add your words
```

### 🎨 Implement Custom Cleaners
Create new cleaning strategies without breaking the pipeline:
```python
class AIGenerativeFillCleaner(CleanStrategy):
    def clean(self, image, bbox):
        # Your AI inpainting logic
        return cleaned_image
```

---

## 🐛 Troubleshooting

### Issue: Thai characters appear overlapped or misaligned
**Solution:** Ensure you're using a quality Thai font that supports combining characters.  
Recommended fonts:
- Thai Sarabun New (Windows default)
- Noto Sans Thai
- Prompt Regular

### Issue: OCR clustering merges unrelated text boxes
**Solution:** Decrease `vertical_gap` threshold or increase `search_window` in `ContentAwareSlicer`.

### Issue: Generated translations are robotic/unnatural
**Solution:** Use Phase 1.5 Human Review to refine translations. Consider upgrading to GPT-4o for better context.

### Issue: `cv2.inpaint()` leaves visible artifacts
**Solution:** Switch to `SolidFillCleaner` for cleaner (though less sophisticated) text removal.

---

## 📊 Performance Metrics

| Operation           | Time (Sample Image 800x15993px) | Notes                            |
| ------------------- | ------------------------------- | -------------------------------- |
| Image Slicing       | ~2-3s                           | ContentAwareSlicer with 14 tiles |
| OCR Extraction      | ~15-30s                         | Depends on image complexity      |
| Translation         | ~5-10s                          | Batch processing via Google API  |
| Typesetting         | ~1-2s                           | Per tile rendering               |
| **Total (Phase 1)** | ~25-45s                         | First run with human pause       |
| **Total (Phase 2)** | ~3-5s                           | Rerun with cached translations   |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

**How to contribute:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Submit a Pull Request

**Ideas for contribution:**
- [ ] Web UI for HITL workflow
- [ ] Support for other language pairs (Japanese→Thai, Korean→Thai)
- [ ] Batch processing mode
- [ ] Discord Bot integration
- [ ] Advanced AI-based text removal (Generative Fill)

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **EasyOCR** for robust multi-language OCR
- **PyThaiNLP** for Thai language processing
- **OpenCV** and **Pillow** for image manipulation
- **Deep-Translator** for seamless translation API integration

---

## 📞 Support

Found a bug or have a question? 
- Open an [Issue](https://github.com/YourUsername/converter-ocr/issues)
- Discussions coming soon!

**Happy translating!** 🎨✨

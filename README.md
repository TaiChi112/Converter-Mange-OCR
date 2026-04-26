>[!WARNING]
>Unpolished & Unorganized: You will likely encounter raw notes, inconsistent Markdown formatting, and incomplete setup guides. Formatting cleanup is performed in batches rather than continuously.

>[!CAUTION]
>Breaking Changes: Expect frequent refactors that may break existing functionality without notice. Do not rely on this project for production-critical tasks until a stable release (v1.0.0) is tagged.

>[!NOTE]
>Project Context: This repository is part of my personal R&D and academic journey as a Computer Science Student interested in becoming an Agentic Software Engineer. It serves as a sandbox for exploring new architectures and performance-oriented languages.

>[!TIP]
>How to Navigate: Since the documentation might lag behind, the most reliable way to understand the current logic is to check the recent commit history or explore the /src directory directly.

>[!IMPORTANT]
>Documentation Mismatch: I am prioritizing the use of AI Assistants for rapid feature iteration over manual documentation updates. Most of the README content below is stale or reflects an older architectural version.

# 💬 Manga Auto-Translator & Smart Thai Typesetter

An end-to-end Python pipeline for translating manga/comics from English to Thai with Human-in-the-Loop (HITL) workflow and smart Thai typesetting.

---

## 📚 Documentation by Language

| Language             | Full Documentation                     |
| -------------------- | -------------------------------------- |
| 🇬🇧 **English**        | [docs/README.md](docs/README.md)       |
| 🇹🇭 **Thai (ภาษาไทย)** | [docs/README_TH.md](docs/README_TH.md) |

---

## ⚡ Quick Start (30 seconds)

```bash
# 1. Setup Python environment
uv python install 3.11
uv venv --python 3.11
uv sync

# 2. Add your Thai font (e.g., tahoma.ttf) to project root

# 3. Place manga image in ./image/ folder and run:
uv run main.py

# 4. Edit translations.json with your preferred translations (Phase 1.5)

# 5. Run again:
uv run main.py
# Output: result_final_pipeline.png and result_step*.png
```

---

## 🌟 Key Features

- **✂️ Content-Aware Slicing** - Intelligently cuts long webtoon strips without slicing through text
- **🧠 Smart OCR Clustering** - Groups related text boxes into paragraphs (not just lines)
- **⏸️ Human-in-the-Loop** - Pause for human review after AI extraction before final rendering
- **🪄 Seamless Inpainting** - Cleanly removes original text with OpenCV Telea algorithm
- **🇹🇭 Thai Tone Mark Shifting** - Pixel-perfect Thai typography without external C-libraries

---

## 📁 Project Structure

```
converter-ocr/
├── main.py                  # Main pipeline
├── pyproject.toml          # Dependencies
├── .gitignore              # Git ignore rules
├── README.md               # You are here
├── docs/
│   ├── README.md           # Full English documentation
│   └── README_TH.md        # Full Thai documentation
├── image/                  # Input manga pages
├── output_tiles/           # Generated tiles
└── translations.json       # HITL checkpoint
```

---

## 🛠️ Tech Stack

```
Python 3.11 | OpenCV | EasyOCR | Deep-Translator | PyThaiNLP | Pillow
```

---

## 📖 Detailed Documentation

**👉 For complete setup, customization, troubleshooting, and API details:**
- **[English Version](docs/README.md)** - Full feature guide, extensibility, and best practices
- **[Thai Version (ภาษาไทย)](docs/README_TH.md)** - เอกสารฉบับภาษาไทยแบบครบถ้วน

---

## 🤝 Contributing

Contributions welcome! Ideas for contribution:
- Web UI for HITL workflow
- Multi-language support (Japanese→Thai, Korean→Thai)
- Batch processing mode
- Discord Bot integration

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🚀 Made with ❤️ for manga/webtoon translators

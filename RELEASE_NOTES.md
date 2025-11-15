# Release Notes

## V1

- **Document ingestion overhaul** – Support PDFs via PyMuPDF + OCR fallback, DOC/DOCX through Mammoth, PPT/PPTX with python-pptx, Excel/ODS via pandas, and standalone image scans with tunable OCR thresholds.
- **OCR controls** – New `ocr_conf_threshold`, `ocr_image_conf_threshold`, language toggles, and two-pass Tesseract flow let you balance noise vs. recall per source.
- **Launchers & config** – `launch_chat_openai.sh` wraps the OpenAI chat bridge; platform-specific requirements files ensure dependency parity; `convert_docs` tasks cover all supported formats out of the box.
- **Hybrid retrieval UX** – Incremental ingestion stays the default, and `rag/chat.py` / `rag/chat_openai.py` support `--attach` for ad hoc context plus OpenAI-compatible generation.

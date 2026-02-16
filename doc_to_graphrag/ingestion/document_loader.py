"""Unified document loader with automatic OCR detection."""

from pathlib import Path
from typing import Dict, Any
import logging

from .pdf_extractor import PDFExtractor
from .ocr_engine import OCREngine

logger = logging.getLogger(__name__)

# File extensions recognised by each handler
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
_PDF_EXTENSIONS = {".pdf"}
_TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml"}


class DocumentLoader:
    """
    Unified document loading with automatic format detection and OCR.

    Decision logic
    ──────────────
    • Image files  → always OCR
    • PDF files    → try pdfplumber first; fall back to OCR if little/no text
    • Text files   → read directly

    Usage:
        loader = DocumentLoader()
        result = loader.load("report.pdf")
        print(result["text"][:200])
        print(result["metadata"])
    """

    def __init__(self, ocr_min_confidence: float = 0.6):
        self.pdf_extractor = PDFExtractor()
        self.ocr_engine = OCREngine(min_confidence=ocr_min_confidence)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def load(self, file_path: str, force_ocr: bool = False) -> Dict[str, Any]:
        """
        Load a document and return its text + metadata.

        Args:
            file_path: Path to the document
            force_ocr:  Force OCR even for native-text PDFs

        Returns:
            {
                "text": str,
                "metadata": {
                    "file_path": str,
                    "file_type": "pdf" | "image" | "text",
                    "page_count": int | None,
                    "ocr_used": bool,
                    "ocr_confidence": float | None,
                }
            }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix in _PDF_EXTENSIONS:
            return self._load_pdf(file_path, force_ocr)
        elif suffix in _IMAGE_EXTENSIONS:
            return self._load_image(file_path)
        elif suffix in _TEXT_EXTENSIONS:
            return self._load_text(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: PDF, images ({', '.join(_IMAGE_EXTENSIONS)}), "
                f"text ({', '.join(_TEXT_EXTENSIONS)})"
            )

    # ------------------------------------------------------------------ #
    #  Private handlers                                                   #
    # ------------------------------------------------------------------ #

    def _load_pdf(self, file_path: str, force_ocr: bool) -> Dict[str, Any]:
        """Load a PDF — native extraction first, OCR as fallback."""
        metadata = self.pdf_extractor.get_metadata(file_path)

        needs_ocr = force_ocr or not self.pdf_extractor.has_extractable_text(file_path)

        if needs_ocr:
            logger.info(f"Using OCR for: {file_path}")
            ocr_result = self.ocr_engine.extract_from_pdf(file_path)
            metadata["ocr_used"] = True
            metadata["ocr_confidence"] = ocr_result["confidence"]
            text = ocr_result["text"]
        else:
            logger.info(f"Using pdfplumber (native text) for: {file_path}")
            text = self.pdf_extractor.extract_text(file_path)
            metadata["ocr_used"] = False
            metadata["ocr_confidence"] = None

        return {"text": text, "metadata": metadata}

    def _load_image(self, file_path: str) -> Dict[str, Any]:
        """Load an image file via OCR."""
        logger.info(f"Using OCR for image: {file_path}")
        ocr_result = self.ocr_engine.extract_from_image(file_path)

        return {
            "text": ocr_result["text"],
            "metadata": {
                "file_path": str(Path(file_path).resolve()),
                "file_type": "image",
                "page_count": 1,
                "ocr_used": True,
                "ocr_confidence": ocr_result["confidence"],
            },
        }

    def _load_text(self, file_path: str) -> Dict[str, Any]:
        """Load a plain text file directly."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return {
            "text": text,
            "metadata": {
                "file_path": str(Path(file_path).resolve()),
                "file_type": "text",
                "page_count": None,
                "ocr_used": False,
                "ocr_confidence": None,
            },
        }

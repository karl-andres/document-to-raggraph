"""OCR engine for scanned documents using pytesseract."""

import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
from typing import Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR engine for extracting text from scanned documents and images."""

    def __init__(self, min_confidence: float = 0.6, lang: str = "eng"):
        """
        Initialize OCR engine.

        Args:
            min_confidence: Minimum confidence threshold (0.0–1.0).
                            Pages below this are flagged as low quality.
            lang: Tesseract language code (default: English)
        """
        self.min_confidence = min_confidence
        self.lang = lang

    # ------------------------------------------------------------------ #
    #  Core: extract from a single image                                  #
    # ------------------------------------------------------------------ #

    def extract_from_image(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Extract text from an image using Tesseract OCR.

        Args:
            image_input: File path (str/Path) or a PIL Image object

        Returns:
            {"text": str, "confidence": float}
            confidence is 0.0–1.0 (average word-level confidence)
        """
        try:
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input)
            else:
                image = image_input

            # Optional preprocessing for better OCR quality
            image = self.preprocess_image(image)

            # Get word-level data with confidence scores
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                output_type=pytesseract.Output.DICT,
            )

            # Calculate average confidence (ignore -1 which means "no text detected")
            confidences = [
                int(c) for c in ocr_data["conf"] if str(c) != "-1"
            ]
            avg_confidence = (
                sum(confidences) / len(confidences) / 100.0
                if confidences
                else 0.0
            )

            # Full text extraction (cleaner than reconstructing from data dict)
            text = pytesseract.image_to_string(image, lang=self.lang)

            return {
                "text": text.strip(),
                "confidence": round(avg_confidence, 4),
            }

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {"text": "", "confidence": 0.0}

    # ------------------------------------------------------------------ #
    #  PDF → images → OCR                                                 #
    # ------------------------------------------------------------------ #

    def extract_from_pdf(self, file_path: str, dpi: int = 300) -> Dict[str, Any]:
        """
        Convert a PDF to images and run OCR on each page.

        Requires `pdf2image` and system-level Poppler installed:
            macOS  → brew install poppler
            Ubuntu → sudo apt install poppler-utils

        Args:
            file_path: Path to the PDF file
            dpi: Resolution for the PDF→image conversion (higher = better OCR, slower)

        Returns:
            {"text": str, "confidence": float, "page_count": int, "per_page": list}
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "pdf2image is required for OCR on PDFs. "
                "Install it with: pip install pdf2image\n"
                "You also need Poppler: brew install poppler (macOS)"
            )

        images = convert_from_path(file_path, dpi=dpi)

        per_page: List[Dict[str, Any]] = []
        all_texts: List[str] = []
        all_confidences: List[float] = []

        for i, img in enumerate(images):
            result = self.extract_from_image(img)
            per_page.append({
                "page": i + 1,
                "text": result["text"],
                "confidence": result["confidence"],
            })
            if result["text"]:
                all_texts.append(result["text"])
            if result["confidence"] > 0:
                all_confidences.append(result["confidence"])

        combined_text = "\n\n".join(all_texts)
        avg_confidence = (
            sum(all_confidences) / len(all_confidences)
            if all_confidences
            else 0.0
        )

        return {
            "text": combined_text,
            "confidence": round(avg_confidence, 4),
            "page_count": len(images),
            "per_page": per_page,
        }

    # ------------------------------------------------------------------ #
    #  Image preprocessing                                                #
    # ------------------------------------------------------------------ #

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply basic preprocessing to improve OCR accuracy.

        Steps:
        1. Convert to grayscale
        2. Light sharpening

        Args:
            image: PIL Image

        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale if not already
        if image.mode != "L":
            image = ImageOps.grayscale(image)

        # Sharpen slightly to help with blurry scans
        image = image.filter(ImageFilter.SHARPEN)

        return image

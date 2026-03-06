"""OCR engine for scanned documents using pytesseract with VLM fallback."""

import base64
import io
import os

import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
from typing import Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR engine for extracting text from scanned documents and images.

    If pytesseract confidence falls below `min_confidence`, the image is
    automatically sent to Mistral OCR 3 via `process_with_ai` for
    higher-quality extraction (handwriting, complex layouts, etc.).
    """

    def __init__(self, min_confidence: float = 0.6, lang: str = "eng"):
        """
        Initialize OCR engine.

        Args:
            min_confidence: Minimum confidence threshold (0.0–1.0).
                            Pages below this are routed to VLM fallback.
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

        If the average word-level confidence is below `min_confidence`,
        the original image is forwarded to `process_with_ai` instead.

        Args:
            image_input: File path (str/Path) or a PIL Image object

        Returns:
            {"text": str, "confidence": float, "ai_extracted": bool}
            confidence is 0.0–1.0 (average word-level confidence)
        """
        try:
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input)
            else:
                image = image_input

            # Keep original for potential VLM fallback (before grayscale)
            original_image = image.copy()

            # Preprocessing for better OCR quality
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

            # If confidence is below threshold, fallback to VLM
            if avg_confidence < self.min_confidence:
                logger.info(
                    f"OCR confidence {avg_confidence:.2%} is below "
                    f"threshold {self.min_confidence:.2%} — falling back to AI extraction"
                )
                return self.process_with_ai(original_image)

            # Full text extraction (cleaner than reconstructing from data dict)
            text = pytesseract.image_to_string(image, lang=self.lang)

            return {
                "text": text.strip(),
                "confidence": round(avg_confidence, 4),
                "ai_extracted": False,
            }

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {"text": "", "confidence": 0.0, "ai_extracted": False}

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
                "ai_extracted": result.get("ai_extracted", False),
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
    #  AI-powered extraction (Mistral OCR fallback)                        #
    # ------------------------------------------------------------------ #

    def process_with_ai(self, image: Image.Image) -> Dict[str, Any]:
        """
        Send an image to Mistral OCR 3 for high-quality text extraction.

        Used as a fallback when pytesseract confidence is below the
        minimum threshold — e.g. handwritten text, degraded scans,
        complex layouts that trip up traditional OCR.

        Mistral OCR 3 is purpose-built for document extraction and
        preserves table structure (HTML with colspan), reading order,
        and handles handwriting natively. Cost: ~$2 / 1,000 pages.

        Requires MISTRAL_API_KEY in environment variables.

        Args:
            image: PIL Image (original, pre-preprocessing)

        Returns:
            {"text": str, "confidence": float, "ai_extracted": True}
        """
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "mistralai is required for AI-based text extraction. "
                "Install it with: pip install mistralai"
            )

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY environment variable is not set. "
                "Get your key at https://console.mistral.ai/"
            )

        # Encode image to base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_data_url = f"data:image/png;base64,{b64_image}"

        client = Mistral(api_key=api_key)

        try:
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": image_data_url,
                },
            )

            # Combine all pages/sections of the OCR result into text
            extracted_parts = []
            for page in ocr_response.pages:
                if page.markdown:
                    extracted_parts.append(page.markdown)

            extracted_text = "\n\n".join(extracted_parts).strip()

            logger.info(
                f"Mistral OCR extraction successful — {len(extracted_text)} chars extracted"
            )

            print("Mistral Used")
            return {
                "text": extracted_text,
                "confidence": 1.0,  # Mistral OCR; confidence is implicit
                "ai_extracted": True,
            }

        except Exception as e:
            logger.error(f"Mistral OCR extraction failed: {e}")
            return {"text": "", "confidence": 0.0, "ai_extracted": True}

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
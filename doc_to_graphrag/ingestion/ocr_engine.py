"""OCR engine for scanned documents using pytesseract with Azure-hosted Mistral OCR fallback.

This implementation mirrors the provided curl examples by:
- Encoding PDFs/images as base64 data URLs
- Calling the Azure-hosted Mistral OCR endpoint via HTTP
- Using AZURE_API_KEY for authentication
"""

import base64
import io
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Literal, Optional
from urllib import request as http_request, error as http_error

import pytesseract
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

OcrMode = Literal["tesseract", "mistral", "hybrid"]
TableFormat = Literal["markdown", "html"]  # None means inline/inline markdown


class OCREngine:
    """OCR engine for extracting text from scanned documents and images.

    Supports three modes:
    - tesseract: Pytesseract only; never call Azure OCR.
    - mistral: Always use Azure-hosted Mistral OCR endpoint (PDF or image).
    - hybrid: Tesseract first; if confidence < min_confidence, fallback to Azure OCR.
    """

    def __init__(
        self,
        mode: OcrMode = "hybrid",
        min_confidence: float = 0.6,
        lang: str = "eng",
        *,
        table_format: Optional[TableFormat] = "markdown",
        extract_header: bool = False,
        extract_footer: bool = False,
        include_image_base64: bool = False,
        # Backwards-compat: accept but ignore legacy mistral_client argument
        mistral_client: Optional[Any] = None,
    ):
        """
        Initialize OCR engine.

        Args:
            mode: "tesseract" | "mistral" | "hybrid". Hybrid uses Azure OCR when
                  Tesseract confidence is below min_confidence.
            min_confidence: Minimum confidence threshold (0.0–1.0) for hybrid mode.
            lang: Tesseract language code (default: English).
            table_format: OCR table output: "markdown" | "html" | None.
            extract_header: Azure OCR: extract header into separate field.
            extract_footer: Azure OCR: extract footer into separate field.
            include_image_base64: Azure OCR: include extracted images as base64.
            mistral_client: legacy parameter from previous Mistral SDK-based
                            implementation; accepted for backwards compatibility
                            but ignored when using the Azure HTTP endpoint.
        """
        self.mode = mode
        self.min_confidence = min_confidence
        self.lang = lang
        self.table_format = table_format
        self.extract_header = extract_header
        self.extract_footer = extract_footer
        self.include_image_base64 = include_image_base64
        # Store but do not use the legacy client to avoid TypeError in
        # existing code that still passes this argument.
        self._legacy_mistral_client = mistral_client

        # Azure-hosted Mistral OCR endpoint + config
        self.azure_endpoint = os.environ.get("AZURE_MISTRAL_OCR_ENDPOINT")
        self.azure_model = os.getenv("AZURE_MISTRAL_OCR_MODEL", "mistral-wizonix-ocr")

    def _mistral_ocr_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for Azure-hosted Mistral OCR from instance options."""
        kwargs: Dict[str, Any] = {}
        if self.table_format is not None:
            kwargs["table_format"] = self.table_format
        if self.extract_header:
            kwargs["extract_header"] = True
        if self.extract_footer:
            kwargs["extract_footer"] = True
        if self.include_image_base64:
            kwargs["include_image_base64"] = True
        return kwargs

    def _call_azure_ocr(self, *, document_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the Azure-hosted Mistral OCR endpoint using the same payload
        structure as the curl examples.

        Requires AZURE_API_KEY in the environment.
        """
        api_key = os.getenv("AZURE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "AZURE_API_KEY environment variable is not set. "
                "Set it to your Azure Mistral OCR API key."
            )

        body = {
            "model": self.azure_model,
            "document": document_payload,
            **self._mistral_ocr_kwargs(),
        }
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        req = http_request.Request(
            self.azure_endpoint,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with http_request.urlopen(req) as resp:
                resp_body = resp.read().decode("utf-8")
                return json.loads(resp_body)
        except http_error.HTTPError as e:
            logger.error(f"Azure Mistral OCR HTTP error: {e.status} {e.reason}")
            raise
        except http_error.URLError as e:
            logger.error(f"Azure Mistral OCR URL error: {e.reason}")
            raise
        except Exception as e:
            logger.error(f"Azure Mistral OCR request failed: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  Core: extract from a single image                                  #
    # ------------------------------------------------------------------ #

    def extract_from_image(
        self, image_input: Union[str, Path, Image.Image], mode: Optional[OcrMode] = None
    ) -> Dict[str, Any]:
        """
        Extract text from an image. Behavior depends on mode:
        - tesseract: Pytesseract only.
        - mistral: Always Azure OCR (skip Tesseract).
        - hybrid: Tesseract first; fallback to Azure OCR if confidence < min_confidence.

        Args:
            image_input: File path (str/Path) or a PIL Image object
            mode: Override instance mode for this call (default: use self.mode)

        Returns:
            {"text": str, "confidence": float, "ai_extracted": bool, "mistral_ocr_response": ...}
        """
        use_mode = mode if mode is not None else self.mode
        try:
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input)
            else:
                image = image_input

            original_image = image.copy()
            image = self.preprocess_image(image)

            # Mistral-only mode: skip Tesseract
            if use_mode == "mistral":
                return self.process_with_ai(original_image)

            # Tesseract path (tesseract mode or hybrid first step)
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                output_type=pytesseract.Output.DICT,
            )
            confidences = [
                int(c) for c in ocr_data["conf"] if str(c) != "-1"
            ]
            avg_confidence = (
                sum(confidences) / len(confidences) / 100.0
                if confidences
                else 0.0
            )

            # Extract text from image_to_data result — avoids a second Tesseract call
            text = " ".join(
                word for word, conf in zip(ocr_data["text"], ocr_data["conf"])
                if str(conf) != "-1" and word.strip()
            )

            # Tesseract-only mode: never fallback
            if use_mode == "tesseract":
                return {
                    "text": text.strip(),
                    "mistral_ocr_response": None,
                    "confidence": round(avg_confidence, 4),
                    "ai_extracted": False,
                }

            # Hybrid: fallback to Azure OCR if below threshold
            if avg_confidence < self.min_confidence:
                logger.info(
                    f"OCR confidence {avg_confidence:.2%} is below "
                    f"threshold {self.min_confidence:.2%} — falling back to Azure OCR"
                )
                return self.process_with_ai(original_image)

            return {
                "text": text.strip(),
                "mistral_ocr_response": None,
                "confidence": round(avg_confidence, 4),
                "ai_extracted": False,
            }

        except (RuntimeError, ImportError):
            raise
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {"text": "", "mistral_ocr_response": None, "confidence": 0.0, "ai_extracted": False}

    # ------------------------------------------------------------------ #
    #  PDF OCR                                                             #
    # ------------------------------------------------------------------ #

    def extract_from_pdf(
        self,
        file_path: str,
        *,
        force_ai: bool = False,
        mode: Optional[OcrMode] = None,
        dpi: int = 300,
    ) -> Dict[str, Any]:
        """
        Extract text from a PDF.

        When mode is "mistral" or force_ai is True, the entire PDF is sent
        to Azure-hosted Mistral OCR in a single API call. Otherwise each page
        is processed via extract_from_image (Tesseract and/or Azure OCR per
        page depending on mode).
        """
        use_mode = mode if mode is not None else self.mode
        use_mistral_whole = use_mode == "mistral" or force_ai
        if use_mistral_whole:
            return self.process_pdf_with_ai(file_path)

        # ── Tesseract path (page-by-page) ──────────────────────────── #
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
        all_ocr_responses: List[Any] = []
        any_ai_extracted = False

        for i, img in enumerate(images):
            result = self.extract_from_image(img, mode=use_mode)
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
            if result.get("mistral_ocr_response") is not None:
                all_ocr_responses.append(result["mistral_ocr_response"])
            if result.get("ai_extracted"):
                any_ai_extracted = True

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
            "mistral_ocr_response": all_ocr_responses if all_ocr_responses else None,
            "ai_extracted": any_ai_extracted,
        }

    # ------------------------------------------------------------------ #
    #  Direct PDF upload to Azure OCR                                     #
    # ------------------------------------------------------------------ #

    def process_pdf_with_ai(self, file_path: str) -> Dict[str, Any]:
        """
        Send an entire PDF to Azure-hosted Mistral OCR in a single API call.

        This mirrors the curl example using a base64-encoded data URL.
        """
        try:
            with open(file_path, "rb") as f:
                raw = f.read()
        except Exception as e:
            logger.error(f"Failed to read PDF for OCR: {e}")
            return {
                "text": "",
                "mistral_ocr_response": None,
                "confidence": 0.0,
                "ai_extracted": True,
                "page_count": 0,
            }

        b64_pdf = base64.b64encode(raw).decode("utf-8")
        data_url = f"data:application/pdf;base64,{b64_pdf}"

        try:
            ocr_response = self._call_azure_ocr(
                document_payload={
                    "type": "document_url",
                    "document_url": data_url,
                }
            )

            pages = ocr_response.get("pages", []) or []
            pages_text = [
                (p.get("markdown") or p.get("text") or "").strip() for p in pages
            ]
            pages_text = [t for t in pages_text if t]
            extracted_text = "\n\n".join(pages_text).strip()
            page_count = (
                ocr_response.get("usage_info", {}).get("pages_processed")
                or len(pages)
            )

            logger.info(
                f"Azure Mistral OCR (PDF) successful — "
                f"{page_count} pages, {len(extracted_text)} chars extracted"
            )

            return {
                "text": extracted_text,
                "mistral_ocr_response": ocr_response,
                "confidence": 1.0,
                "ai_extracted": True,
                "page_count": page_count,
            }

        except Exception as e:
            logger.error(f"Azure Mistral OCR (PDF) failed: {e}")
            return {
                "text": "",
                "mistral_ocr_response": None,
                "confidence": 0.0,
                "ai_extracted": True,
                "page_count": 0,
            }

    # ------------------------------------------------------------------ #
    #  AI-powered extraction (Azure OCR fallback)                         #
    # ------------------------------------------------------------------ #

    def process_with_ai(self, image: Image.Image) -> Dict[str, Any]:
        """
        Send an image to Azure-hosted Mistral OCR for high-quality text extraction.

        This mirrors the curl example using a base64-encoded image data URL.
        """
        # Encode image to base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_data_url = f"data:image/png;base64,{b64_image}"

        try:
            ocr_response = self._call_azure_ocr(
                document_payload={
                    "type": "image_url",
                    "image_url": image_data_url,
                }
            )

            pages = ocr_response.get("pages", []) or []
            extracted_parts: List[str] = []
            for page in pages:
                md = (page.get("markdown") or page.get("text") or "").strip()
                if md:
                    extracted_parts.append(md)

            extracted_text = "\n\n".join(extracted_parts).strip()

            logger.info(
                f"Azure Mistral OCR (image) successful — {len(extracted_text)} chars extracted"
            )

            return {
                "text": extracted_text,
                "mistral_ocr_response": ocr_response,
                "confidence": 1.0,  # AI OCR; confidence treated as max
                "ai_extracted": True,
            }

        except Exception as e:
            logger.error(f"Azure Mistral OCR (image) failed: {e}")
            return {
                "text": "",
                "mistral_ocr_response": None,
                "confidence": 0.0,
                "ai_extracted": True,
            }

    # ------------------------------------------------------------------ #
    #  Mistral Batch OCR (not supported with Azure endpoint)             #
    # ------------------------------------------------------------------ #

    def process_batch_mistral(
        self,
        file_paths: List[Union[str, Path]],
    ) -> str:
        """
        Batch OCR is not supported when using the Azure curl-based endpoint.
        """
        raise NotImplementedError(
            "Batch OCR is not supported with the Azure Mistral OCR endpoint. "
            "Call extract_from_pdf/extract_from_image in a loop instead."
        )

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


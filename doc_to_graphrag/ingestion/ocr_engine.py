"""OCR engine for scanned documents using pytesseract with VLM fallback."""

import base64
import io
import os

import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
from typing import Dict, Any, Union, List, Literal, Optional
import logging

logger = logging.getLogger(__name__)

OcrMode = Literal["tesseract", "mistral", "hybrid"]
TableFormat = Literal["markdown", "html"]  # None means inline/inline markdown



class OCREngine:
    """OCR engine for extracting text from scanned documents and images.

    Supports three modes:
    - tesseract: Pytesseract only; never call Mistral.
    - mistral: Always use Mistral OCR API (PDF upload or image base64).
    - hybrid: Tesseract first; if confidence < min_confidence, fallback to Mistral.
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
        mistral_client=None,
    ):
        """
        Initialize OCR engine.

        Args:
            mode: "tesseract" | "mistral" | "hybrid". Hybrid uses Mistral when
                  Tesseract confidence is below min_confidence.
            min_confidence: Minimum confidence threshold (0.0–1.0) for hybrid mode.
            lang: Tesseract language code (default: English).
            table_format: Mistral OCR table output: "markdown" | "html" | None.
            extract_header: Mistral: extract header into separate field.
            extract_footer: Mistral: extract footer into separate field.
            include_image_base64: Mistral: include extracted images as base64.
            mistral_client: Pre-initialized Mistral client. If None, one will be
                            created from MISTRAL_API_KEY when first needed.
        """
        self.mode = mode
        self.min_confidence = min_confidence
        self.lang = lang
        self.table_format = table_format
        self.extract_header = extract_header
        self.extract_footer = extract_footer
        self.include_image_base64 = include_image_base64
        self._mistral_client = mistral_client

    def _mistral_ocr_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for client.ocr.process from instance options."""
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

    def _get_mistral_client(self):
        """Return the Mistral client, creating one from env if not injected."""
        if self._mistral_client is not None:
            return self._mistral_client
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "mistralai is required for AI-based OCR. pip install mistralai"
            )
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY environment variable is not set. "
                "Get your key at https://console.mistral.ai/"
            )
        self._mistral_client = Mistral(api_key=api_key)
        return self._mistral_client

    # ------------------------------------------------------------------ #
    #  Core: extract from a single image                                  #
    # ------------------------------------------------------------------ #

    def extract_from_image(
        self, image_input: Union[str, Path, Image.Image], mode: Optional[OcrMode] = None
    ) -> Dict[str, Any]:
        """
        Extract text from an image. Behavior depends on mode:
        - tesseract: Pytesseract only.
        - mistral: Always Mistral OCR (skip Tesseract).
        - hybrid: Tesseract first; fallback to Mistral if confidence < min_confidence.

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

            # Hybrid: fallback to Mistral if below threshold
            if avg_confidence < self.min_confidence:
                logger.info(
                    f"OCR confidence {avg_confidence:.2%} is below "
                    f"threshold {self.min_confidence:.2%} — falling back to AI extraction"
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

        When mode is "mistral" or force_ai is True, the entire PDF is uploaded
        to Mistral OCR in a single API call. Otherwise each page is processed
        via extract_from_image (Tesseract and/or Mistral per page depending on mode).

        Args:
            file_path:  Path to the PDF file
            force_ai:   When in hybrid mode, skip Tesseract and use Mistral for
                        the whole document. Ignored when mode is tesseract or mistral.
            mode:      Override instance mode for this call.
            dpi:        Resolution for pdf2image (only when not using Mistral for whole doc)

        Returns:
            {"text": str, "confidence": float, "page_count": int,
             "mistral_ocr_response": ..., "ai_extracted": bool}
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
    #  Direct PDF upload to Mistral OCR                                    #
    # ------------------------------------------------------------------ #

    def process_pdf_with_ai(self, file_path: str) -> Dict[str, Any]:
        """
        Upload an entire PDF to Mistral OCR in a single API call.

        This is **much** faster than per-page processing — Mistral can
        handle up to 1 000 pages / 50 MB and processes ~2 000 pages/min.

        Workflow:
            1. Upload the PDF to Mistral file storage
            2. Obtain a signed URL for the uploaded file
            3. Call ``client.ocr.process`` with the signed URL

        Args:
            file_path: Path to the PDF file

        Returns:
            {"text": str, "mistral_ocr_response": OCRResponse,
             "confidence": 1.0, "ai_extracted": True,
             "page_count": int}
        """
        client = self._get_mistral_client()

        try:
            # 1. Upload the PDF to Mistral's file storage
            with open(file_path, "rb") as f:
                uploaded = client.files.upload(
                    file={"file_name": Path(file_path).name, "content": f},
                    purpose="ocr",
                )

            # 2. Get a signed URL for the uploaded file
            signed_url = client.files.get_signed_url(file_id=uploaded.id)

            # 3. Process the entire PDF in one call
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
                **self._mistral_ocr_kwargs(),
            )

            # Combine all pages into a single text
            pages_text = [p.markdown for p in ocr_response.pages if p.markdown]
            extracted_text = "\n\n".join(pages_text).strip()

            logger.info(
                f"Mistral OCR (PDF upload) successful — "
                f"{len(ocr_response.pages)} pages, "
                f"{len(extracted_text)} chars extracted"
            )

            return {
                "text": extracted_text,
                "mistral_ocr_response": ocr_response,
                "confidence": 1.0,
                "ai_extracted": True,
                "page_count": len(ocr_response.pages),
            }

        except Exception as e:
            logger.error(f"Mistral OCR (PDF upload) failed: {e}")
            return {
                "text": "",
                "mistral_ocr_response": None,
                "confidence": 0.0,
                "ai_extracted": True,
                "page_count": 0,
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
        client = self._get_mistral_client()

        # Encode image to base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_data_url = f"data:image/png;base64,{b64_image}"

        try:
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": image_data_url,
                },
                **self._mistral_ocr_kwargs(),
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

            return {
                "text": extracted_text,
                "mistral_ocr_response": ocr_response,
                "confidence": 1.0,  # Mistral OCR; confidence is implicit
                "ai_extracted": True,
            }

        except Exception as e:
            logger.error(f"Mistral OCR extraction failed: {e}")
            return {"text": "", "mistral_ocr_response": None, "confidence": 0.0, "ai_extracted": True}

    # ------------------------------------------------------------------ #
    #  Mistral Batch OCR (Mistral-only)                                   #
    # ------------------------------------------------------------------ #

    def process_batch_mistral(
        self,
        file_paths: List[Union[str, Path]],
    ) -> str:
        """
        Submit multiple documents to Mistral Batch Inference API and return the job ID.

        Only valid when mode is "mistral". Supports PDF and image files.
        Use the returned job ID to check status and retrieve results later.
        """
        if self.mode != "mistral":
            raise ValueError(
                "Batch processing is only supported when ocr_mode is 'mistral'. "
                f"Current mode is '{self.mode}'."
            )

        import json
        import tempfile

        client = self._get_mistral_client()
        path_list = [Path(p) for p in file_paths]
        ocr_bodies: List[Dict[str, Any]] = []

        for path in path_list:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            suffix = path.suffix.lower()
            body: Dict[str, Any] = {**self._mistral_ocr_kwargs()}
            if suffix == ".pdf":
                with open(path, "rb") as f:
                    uploaded = client.files.upload(
                        file={"file_name": path.name, "content": f},
                        purpose="ocr",
                    )
                signed_url = client.files.get_signed_url(file_id=uploaded.id)
                body["document"] = {"document_url": signed_url.url}
            else:
                with open(path, "rb") as f:
                    raw = f.read()
                b64 = base64.b64encode(raw).decode("utf-8")
                mime = "image/png" if suffix == ".png" else "image/jpeg"
                body["document"] = {"image_url": f"data:{mime};base64,{b64}"}
            ocr_bodies.append(body)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            for idx, body in enumerate(ocr_bodies):
                line = json.dumps({"custom_id": str(idx), "body": body}) + "\n"
                tmp.write(line)
            jsonl_path = tmp.name

        try:
            with open(jsonl_path, "rb") as f:
                batch_file = client.files.upload(
                    file={"file_name": "ocr_batch.jsonl", "content": f},
                    purpose="batch",
                )
            job = client.batch.jobs.create(
                input_files=[batch_file.id],
                model="mistral-ocr-latest",
                endpoint="/v1/ocr",
            )
        finally:
            try:
                os.unlink(jsonl_path)
            except OSError:
                pass

        logger.info(f"Mistral batch job submitted — job_id={job.id}")
        return job.id

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
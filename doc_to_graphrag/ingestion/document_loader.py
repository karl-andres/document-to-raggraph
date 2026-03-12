"""Unified document loader with automatic OCR detection using Azure-hosted Mistral OCR."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import logging

from .pdf_extractor import PDFExtractor
from .ocr_engine import OCREngine, OcrMode

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

    def __init__(
        self,
        ocr_mode: OcrMode = "hybrid",
        ocr_min_confidence: float = 0.6,
    ):
        self.pdf_extractor = PDFExtractor()
        self.ocr_engine = OCREngine(
            mode=ocr_mode,
            min_confidence=ocr_min_confidence,
        )
        self.ocr_mode = ocr_mode

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def load(
        self,
        file_path: str,
        force_ocr: bool = False,
        force_ai: bool = False,
        ocr_mode: Optional[OcrMode] = None,
    ) -> Dict[str, Any]:
        """
        Load a document and return its text + metadata.

        Args:
            file_path: Path to the document
            force_ocr: Force OCR even for native-text PDFs (use OCR instead of pdfplumber)
            force_ai: If True, use Azure OCR for this call (overrides instance ocr_mode for this load)
            ocr_mode: Override instance ocr_mode for this call: "tesseract" | "mistral" | "hybrid"

        Returns:
            {
                "text": str,
                "metadata": {...},
                "mistral_ocr_response": ... (if Azure OCR was used),
                "ai_extracted": bool | None,
            }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Resolve effective OCR mode for this call
        effective_mode: OcrMode = ocr_mode if ocr_mode is not None else self.ocr_mode
        if force_ai:
            effective_mode = "mistral"

        suffix = path.suffix.lower()

        if suffix in _PDF_EXTENSIONS:
            return self._load_pdf(file_path, force_ocr, effective_mode)
        elif suffix in _IMAGE_EXTENSIONS:
            return self._load_image(file_path, effective_mode)
        elif suffix in _TEXT_EXTENSIONS:
            return self._load_text(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: PDF, images ({', '.join(_IMAGE_EXTENSIONS)}), "
                f"text ({', '.join(_TEXT_EXTENSIONS)})"
            )

    def load_batch(
        self,
        file_paths: List[str],
    ) -> str:
        """
        Batch OCR is not supported with the Azure curl-based endpoint.
        This method exists to keep the API surface but will always raise.
        """
        raise NotImplementedError(
            "Batch OCR is not supported with the Azure Mistral OCR endpoint. "
            "Call load() in a loop for multiple files."
        )

    def check_batch(self, job_id: str) -> Dict[str, Any]:
        """
        Placeholder for batch status API — not supported with Azure endpoint.
        """
        raise NotImplementedError("Batch status is not supported with the Azure OCR endpoint.")

    def get_batch(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Placeholder for batch result retrieval — not supported with Azure endpoint.
        """
        raise NotImplementedError("Batch result retrieval is not supported with the Azure OCR endpoint.")

    # ------------------------------------------------------------------ #
    #  Private handlers                                                   #
    # ------------------------------------------------------------------ #

    def _load_pdf(self, file_path: str, force_ocr: bool, ocr_mode: OcrMode) -> Dict[str, Any]:
        """Load a PDF — native extraction first, OCR as fallback."""
        metadata = self.pdf_extractor.get_metadata(file_path)

        needs_ocr = force_ocr or not self.pdf_extractor.has_extractable_text(file_path)

        if needs_ocr:
            logger.info(f"Using OCR for: {file_path}")
            ocr_result = self.ocr_engine.extract_from_pdf(
                file_path, force_ai=(ocr_mode == "mistral"), mode=ocr_mode
            )
            metadata["ocr_used"] = True
            metadata["ocr_confidence"] = ocr_result["confidence"]
            text = ocr_result["text"]
            mistral_ocr_response = ocr_result["mistral_ocr_response"]
            ai_extracted = ocr_result["ai_extracted"]
        else:
            logger.info(f"Using pdfplumber (native text) for: {file_path}")
            text = self.pdf_extractor.extract_text(file_path)
            metadata["ocr_used"] = False
            metadata["ocr_confidence"] = None
            mistral_ocr_response = None
            ai_extracted = False

        return {
            "text": text,
            "mistral_ocr_response": mistral_ocr_response,
            "metadata": metadata,
            "ai_extracted": ai_extracted,
        }

    def _load_image(self, file_path: str, ocr_mode: OcrMode) -> Dict[str, Any]:
        """Load an image file via OCR."""
        logger.info(f"Using OCR for image: {file_path}")
        ocr_result = self.ocr_engine.extract_from_image(file_path, mode=ocr_mode)

        return {
            "text": ocr_result["text"],
            "mistral_ocr_response": ocr_result["mistral_ocr_response"],
            "metadata": {
                "file_path": str(Path(file_path).resolve()),
                "file_type": "image",
                "page_count": 1,
                "ocr_used": True,
                "ocr_confidence": ocr_result["confidence"],
            },
            "ai_extracted": ocr_result["ai_extracted"],
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

"""Unified document loader with automatic OCR detection."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import logging

from .pdf_extractor import PDFExtractor
from .ocr_engine import OCREngine, OcrMode

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

    def __init__(
        self,
        ocr_mode: OcrMode = "hybrid",
        ocr_min_confidence: float = 0.6,
    ):
        self._mistral_client = self._init_mistral_client()
        self.pdf_extractor = PDFExtractor()
        self.ocr_engine = OCREngine(
            mode=ocr_mode,
            min_confidence=ocr_min_confidence,
            mistral_client=self._mistral_client,
        )
        self.ocr_mode = ocr_mode

    def _init_mistral_client(self):
        """Initialize the Mistral client from environment, or return None."""
        try:
            from mistralai.client import Mistral
        except ImportError:
            return None
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return None
        return Mistral(api_key=api_key)

    def _require_mistral_client(self):
        """Return the Mistral client, raising if not available."""
        if self._mistral_client is None:
            raise RuntimeError(
                "Mistral client is not initialized. Ensure mistralai is installed "
                "and MISTRAL_API_KEY is set."
            )
        return self._mistral_client

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def load(
        self,
        file_path: str,
        force_ocr: bool = False,
        force_ai: bool = False,
        ocr_mode: Optional[OcrMode] = None,
    ) -> Dict[str, Any]:
        """
        Load a document and return its text + metadata.

        Args:
            file_path: Path to the document
            force_ocr: Force OCR even for native-text PDFs (use OCR instead of pdfplumber)
            force_ai: If True, use Mistral OCR for this call (overrides instance ocr_mode for this load)
            ocr_mode: Override instance ocr_mode for this call: "tesseract" | "mistral" | "hybrid"

        Returns:
            {
                "text": str,
                "metadata": {...},
                "mistral_ocr_response": ... (if Mistral was used),
                "ai_extracted": bool | None,
            }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Resolve effective OCR mode for this call
        effective_mode: OcrMode = ocr_mode if ocr_mode is not None else self.ocr_mode
        if force_ai:
            effective_mode = "mistral"

        suffix = path.suffix.lower()

        if suffix in _PDF_EXTENSIONS:
            return self._load_pdf(file_path, force_ocr, effective_mode)
        elif suffix in _IMAGE_EXTENSIONS:
            return self._load_image(file_path, effective_mode)
        elif suffix in _TEXT_EXTENSIONS:
            return self._load_text(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: PDF, images ({', '.join(_IMAGE_EXTENSIONS)}), "
                f"text ({', '.join(_TEXT_EXTENSIONS)})"
            )

    def load_batch(
        self,
        file_paths: List[str],
    ) -> str:
        """
        Submit multiple documents to Mistral Batch Inference API and return the job ID.

        Only available when ocr_mode is "mistral". For Tesseract or Hybrid mode,
        call load() in a loop for multiple files.

        Use the returned job ID with check_batch() to poll status.

        Args:
            file_paths: Paths to PDF or image files.

        Returns:
            Mistral batch job ID string.

        Raises:
            ValueError: If ocr_mode is not "mistral".
        """
        if self.ocr_mode != "mistral":
            raise ValueError(
                "Batch loading is only supported when ocr_mode is 'mistral'. "
                f"Current mode is '{self.ocr_mode}'. For multiple files with "
                "Tesseract or Hybrid, call load() in a loop."
            )
        return self.ocr_engine.process_batch_mistral(file_paths)

    def check_batch(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a Mistral batch OCR job.

        Args:
            job_id: The batch job ID returned by load_batch().

        Returns:
            {
                "job_id": str,
                "status": str,  # QUEUED | RUNNING | SUCCESS | FAILED |
                                #  TIMEOUT_EXCEEDED | CANCELLATION_REQUESTED | CANCELLED
                "done": bool,   # True when status is terminal (not QUEUED or RUNNING)
            }
        """
        client = self._require_mistral_client()
        job = client.batch.jobs.get(job_id=job_id)
        status = getattr(job, "status", None)
        done = status not in ("QUEUED", "RUNNING")

        return {
            "job_id": job_id,
            "status": status,
            "done": done,
        }

    def get_batch(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Download and parse results for a completed Mistral batch OCR job.

        Call only after check_batch() returns done=True and status="SUCCESS".
        Results are returned in the same order as the original load_batch() call.

        Returns:
            List of dicts matching load() shape:
            {
                "text": str,
                "metadata": {"page_count": int, "ocr_used": True, "ocr_confidence": float},
                "mistral_ocr_response": dict,
                "ai_extracted": True,
                "error": str,  # only present on per-item failure
            }
            Note: "file_path" and "file_type" are omitted — not available in batch results.
        """
        client = self._require_mistral_client()
        job = client.batch.jobs.get(job_id=job_id)
        output_file_stream = client.files.download(file_id=job.output_file)
        lines = [
            json.loads(line)
            for line in output_file_stream.read().decode("utf-8").splitlines()
            if line.strip()
        ]

        # Restore original submission order via custom_id
        lines.sort(key=lambda x: int(x.get("custom_id", 0)))

        results = []
        for entry in lines:
            error = entry.get("error")
            response = entry.get("response", {})
            status_code = response.get("status_code")
            body = response.get("body", {})

            if error or status_code != 200:
                results.append({
                    "text": "",
                    "metadata": {"page_count": 0, "ocr_used": True, "ocr_confidence": 0.0},
                    "mistral_ocr_response": body or None,
                    "ai_extracted": True,
                    "error": error or f"HTTP {status_code}",
                })
                continue

            pages = body.get("pages", [])
            text = "\n\n".join(p["markdown"] for p in pages if p.get("markdown")).strip()
            page_count = body.get("usage_info", {}).get("pages_processed", len(pages))

            results.append({
                "text": text,
                "metadata": {"page_count": page_count, "ocr_used": True, "ocr_confidence": 1.0},
                "mistral_ocr_response": body,
                "ai_extracted": True,
            })

        return results

    # ------------------------------------------------------------------ #
    #  Private handlers                                                   #
    # ------------------------------------------------------------------ #

    def _load_pdf(self, file_path: str, force_ocr: bool, ocr_mode: OcrMode) -> Dict[str, Any]:
        """Load a PDF — native extraction first, OCR as fallback."""
        metadata = self.pdf_extractor.get_metadata(file_path)

        needs_ocr = force_ocr or not self.pdf_extractor.has_extractable_text(file_path)

        if needs_ocr:
            logger.info(f"Using OCR for: {file_path}")
            ocr_result = self.ocr_engine.extract_from_pdf(
                file_path, force_ai=(ocr_mode == "mistral"), mode=ocr_mode
            )
            metadata["ocr_used"] = True
            metadata["ocr_confidence"] = ocr_result["confidence"]
            text = ocr_result["text"]
            mistral_ocr_response = ocr_result["mistral_ocr_response"]
            ai_extracted = ocr_result["ai_extracted"]
        else:
            logger.info(f"Using pdfplumber (native text) for: {file_path}")
            text = self.pdf_extractor.extract_text(file_path)
            metadata["ocr_used"] = False
            metadata["ocr_confidence"] = None
            mistral_ocr_response = None
            ai_extracted = False

        return {
            "text": text,
            "mistral_ocr_response": mistral_ocr_response,
            "metadata": metadata,
            "ai_extracted": ai_extracted,
        }

    def _load_image(self, file_path: str, ocr_mode: OcrMode) -> Dict[str, Any]:
        """Load an image file via OCR."""
        logger.info(f"Using OCR for image: {file_path}")
        ocr_result = self.ocr_engine.extract_from_image(file_path, mode=ocr_mode)

        return {
            "text": ocr_result["text"],
            "mistral_ocr_response": ocr_result["mistral_ocr_response"],
            "metadata": {
                "file_path": str(Path(file_path).resolve()),
                "file_type": "image",
                "page_count": 1,
                "ocr_used": True,
                "ocr_confidence": ocr_result["confidence"],
            },
            "ai_extracted": ocr_result["ai_extracted"],
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

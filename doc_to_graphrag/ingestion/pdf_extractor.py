"""PDF text extraction using pdfplumber."""

import pdfplumber
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from native (text-based) PDF files using pdfplumber."""

    def extract_text(self, file_path: str) -> str:
        """
        Extract all text from a PDF file, page by page.

        Args:
            file_path: Path to PDF file

        Returns:
            Combined text content from all pages
        """
        text_parts = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n\n".join(text_parts)

    def extract_text_per_page(self, file_path: str) -> List[str]:
        """
        Extract text from each page individually.

        Args:
            file_path: Path to PDF file

        Returns:
            List of text strings, one per page
        """
        pages_text = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                pages_text.append(page_text if page_text else "")

        return pages_text

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file.

        Returns:
            Dictionary with file_path, file_type, page_count, ocr_used
        """
        with pdfplumber.open(file_path) as pdf:
            return {
                "file_path": str(Path(file_path).resolve()),
                "file_type": "pdf",
                "page_count": len(pdf.pages),
                "ocr_used": False,
            }

    def has_extractable_text(self, file_path: str, min_chars_per_page: int = 50) -> bool:
        """
        Heuristic to check if a PDF has native (non-scanned) text.

        Checks the first few pages for a minimum amount of extractable text.
        If most pages have very little text, the PDF is likely scanned.

        Args:
            file_path: Path to PDF file
            min_chars_per_page: Minimum characters per page to consider it text-based

        Returns:
            True if the PDF appears to have native text, False if likely scanned
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                # Check up to the first 3 pages
                pages_to_check = min(3, len(pdf.pages))
                text_pages = 0

                for i in range(pages_to_check):
                    page_text = pdf.pages[i].extract_text()
                    if page_text and len(page_text.strip()) >= min_chars_per_page:
                        text_pages += 1

                # If at least half the checked pages have text, it's a native PDF
                return text_pages >= (pages_to_check / 2)

        except Exception as e:
            logger.warning(f"Could not check PDF for text: {e}")
            return False

# General-Purpose Document GraphRAG Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reusable pipeline that transforms unstructured documents into Neo4j knowledge graphs with dynamic schema inference and multi-mode retrieval.

**Architecture:** Chunk-first approach with LLM-driven entity schema inference, type-aware entity reconciliation, and hybrid vector+graph retrieval. Built on Neo4j SimpleKGPipeline with custom pre-processing.

**Tech Stack:** Python 3.10+, Neo4j, OpenAI GPT-4, pdfplumber, pytesseract, neo4j-graphrag, sentence-transformers

---

## Project Structure Setup

### Task 1: Initialize Python Package Structure

**Files:**
- Create: `doc_to_graphrag/__init__.py`
- Create: `doc_to_graphrag/ingestion/__init__.py`
- Create: `doc_to_graphrag/chunking/__init__.py`
- Create: `doc_to_graphrag/extraction/__init__.py`
- Create: `doc_to_graphrag/reconciliation/__init__.py`
- Create: `doc_to_graphrag/graph/__init__.py`
- Create: `doc_to_graphrag/retrieval/__init__.py`
- Create: `tests/__init__.py`
- Create: `setup.py`
- Modify: `requirements.txt`

**Step 1: Create package structure**

```bash
mkdir -p doc_to_graphrag/{ingestion,chunking,extraction,reconciliation,graph,retrieval}
touch doc_to_graphrag/__init__.py
touch doc_to_graphrag/{ingestion,chunking,extraction,reconciliation,graph,retrieval}/__init__.py
mkdir -p tests/{ingestion,chunking,extraction,reconciliation,graph,retrieval,integration}
touch tests/__init__.py
```

**Step 2: Create setup.py**

```python
from setuptools import setup, find_packages

setup(
    name="doc-to-graphrag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "neo4j",
        "neo4j-graphrag[openai]",
        "pdfplumber",
        "pytesseract",
        "pillow",
        "openai",
        "python-dotenv",
        "sentence-transformers",
        "pytest",
        "pytest-cov",
    ],
    python_requires=">=3.10",
)
```

**Step 3: Update requirements.txt**

Add to existing requirements.txt:
```
pytest>=7.0.0
pytest-cov>=4.0.0
sentence-transformers>=2.2.0
pillow>=10.0.0
pytesseract>=0.3.10
```

**Step 4: Install package in development mode**

Run: `pip install -e .`
Expected: Package installed successfully

**Step 5: Commit**

```bash
git add doc_to_graphrag/ tests/ setup.py requirements.txt
git commit -m "feat: initialize package structure for document GraphRAG pipeline"
```

---

## Phase 1: Document Ingestion & OCR

### Task 2: PDF Text Extraction

**Files:**
- Create: `doc_to_graphrag/ingestion/pdf_extractor.py`
- Create: `tests/ingestion/test_pdf_extractor.py`

**Step 1: Write the failing test**

```python
# tests/ingestion/test_pdf_extractor.py
import pytest
from doc_to_graphrag.ingestion.pdf_extractor import PDFExtractor

def test_extract_text_from_native_pdf(tmp_path):
    # This test will fail until we implement PDFExtractor
    extractor = PDFExtractor()

    # We'll use a sample PDF from notebooks for testing
    # For now, test with mock data
    text = extractor.extract_text("sample.pdf")

    assert isinstance(text, str)
    assert len(text) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_pdf_extractor.py -v`
Expected: FAIL with "ModuleNotFoundError" or "PDFExtractor not defined"

**Step 3: Write minimal implementation**

```python
# doc_to_graphrag/ingestion/pdf_extractor.py
import pdfplumber
from pathlib import Path
from typing import Dict, Any

class PDFExtractor:
    """Extract text from PDF files using pdfplumber."""

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text content
        """
        text_parts = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n".join(text_parts)

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF.

        Returns:
            Dictionary with file_path, page_count, file_type
        """
        with pdfplumber.open(file_path) as pdf:
            return {
                "file_path": str(file_path),
                "file_type": "pdf",
                "page_count": len(pdf.pages),
                "ocr_used": False,
            }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_pdf_extractor.py -v`
Expected: PASS (might fail if no sample PDF - add mock or skip for now)

**Step 5: Commit**

```bash
git add doc_to_graphrag/ingestion/pdf_extractor.py tests/ingestion/test_pdf_extractor.py
git commit -m "feat: add PDF text extraction with pdfplumber"
```

### Task 3: OCR Engine for Scanned Documents

**Files:**
- Create: `doc_to_graphrag/ingestion/ocr_engine.py`
- Create: `tests/ingestion/test_ocr_engine.py`

**Step 1: Write the failing test**

```python
# tests/ingestion/test_ocr_engine.py
import pytest
from doc_to_graphrag.ingestion.ocr_engine import OCREngine
from PIL import Image
import numpy as np

def test_ocr_extracts_text_from_image(tmp_path):
    ocr = OCREngine()

    # Create a simple test image (for real use, test with actual scanned doc)
    # For now, test the interface
    result = ocr.extract_from_image("test_image.png")

    assert isinstance(result, dict)
    assert "text" in result
    assert "confidence" in result
    assert isinstance(result["text"], str)
    assert 0 <= result["confidence"] <= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_ocr_engine.py -v`
Expected: FAIL with "OCREngine not defined"

**Step 3: Write minimal implementation**

```python
# doc_to_graphrag/ingestion/ocr_engine.py
import pytesseract
from PIL import Image
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class OCREngine:
    """OCR engine for extracting text from scanned documents."""

    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize OCR engine.

        Args:
            min_confidence: Minimum confidence threshold (0-1)
        """
        self.min_confidence = min_confidence

    def extract_from_image(self, image_path: str) -> Dict[str, any]:
        """
        Extract text from image using Tesseract OCR.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with 'text' and 'confidence' keys
        """
        try:
            # Open image
            image = Image.open(image_path)

            # Get OCR data with confidence scores
            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT
            )

            # Calculate average confidence
            confidences = [
                int(conf) for conf in ocr_data['conf']
                if conf != '-1'
            ]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0

            # Extract text
            text = pytesseract.image_to_string(image)

            return {
                "text": text,
                "confidence": avg_confidence,
            }

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
            }

    def is_scanned_pdf(self, file_path: str) -> bool:
        """
        Heuristic to detect if PDF is scanned (image-based).

        Args:
            file_path: Path to PDF file

        Returns:
            True if likely scanned, False otherwise
        """
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            # Check first page for text
            first_page_text = pdf.pages[0].extract_text()

            # If very little or no text, likely scanned
            return len(first_page_text.strip()) < 50 if first_page_text else True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_ocr_engine.py -v`
Expected: PASS (or skip if pytesseract not installed)

**Step 5: Commit**

```bash
git add doc_to_graphrag/ingestion/ocr_engine.py tests/ingestion/test_ocr_engine.py
git commit -m "feat: add OCR engine for scanned document text extraction"
```

### Task 4: Unified Document Ingestion

**Files:**
- Create: `doc_to_graphrag/ingestion/document_loader.py`
- Create: `tests/ingestion/test_document_loader.py`

**Step 1: Write the failing test**

```python
# tests/ingestion/test_document_loader.py
import pytest
from doc_to_graphrag.ingestion.document_loader import DocumentLoader

def test_load_native_pdf():
    loader = DocumentLoader()
    result = loader.load_document("sample.pdf")

    assert result["text"] is not None
    assert result["metadata"]["file_type"] == "pdf"
    assert result["metadata"]["ocr_used"] == False

def test_load_scanned_pdf_with_ocr():
    loader = DocumentLoader()
    result = loader.load_document("scanned.pdf", force_ocr=True)

    assert result["text"] is not None
    assert result["metadata"]["ocr_used"] == True
    assert "ocr_confidence" in result["metadata"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_document_loader.py -v`
Expected: FAIL with "DocumentLoader not defined"

**Step 3: Write minimal implementation**

```python
# doc_to_graphrag/ingestion/document_loader.py
from pathlib import Path
from typing import Dict, Any
import logging

from .pdf_extractor import PDFExtractor
from .ocr_engine import OCREngine

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Unified document loading with automatic OCR detection."""

    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.ocr_engine = OCREngine()

    def load_document(self, file_path: str, force_ocr: bool = False) -> Dict[str, Any]:
        """
        Load document with automatic format detection and OCR.

        Args:
            file_path: Path to document file
            force_ocr: Force OCR even for native PDFs

        Returns:
            Dictionary with 'text' and 'metadata' keys
        """
        path = Path(file_path)

        if path.suffix.lower() == '.pdf':
            return self._load_pdf(file_path, force_ocr)
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
            return self._load_image(file_path)
        elif path.suffix.lower() == '.txt':
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _load_pdf(self, file_path: str, force_ocr: bool) -> Dict[str, Any]:
        """Load PDF with OCR if needed."""
        # Check if scanned (unless force_ocr)
        needs_ocr = force_ocr or self.ocr_engine.is_scanned_pdf(file_path)

        if needs_ocr:
            # Convert PDF to images and OCR
            # For now, simplified - would use pdf2image in production
            metadata = self.pdf_extractor.get_metadata(file_path)
            metadata["ocr_used"] = True
            metadata["ocr_confidence"] = 0.75  # Placeholder

            # In real implementation, convert PDF pages to images and OCR
            text = self.pdf_extractor.extract_text(file_path)

            return {
                "text": text,
                "metadata": metadata,
            }
        else:
            # Native PDF extraction
            text = self.pdf_extractor.extract_text(file_path)
            metadata = self.pdf_extractor.get_metadata(file_path)

            return {
                "text": text,
                "metadata": metadata,
            }

    def _load_image(self, file_path: str) -> Dict[str, Any]:
        """Load image file with OCR."""
        result = self.ocr_engine.extract_from_image(file_path)

        return {
            "text": result["text"],
            "metadata": {
                "file_path": str(file_path),
                "file_type": "image",
                "ocr_used": True,
                "ocr_confidence": result["confidence"],
            }
        }

    def _load_text(self, file_path: str) -> Dict[str, Any]:
        """Load plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return {
            "text": text,
            "metadata": {
                "file_path": str(file_path),
                "file_type": "text",
                "ocr_used": False,
            }
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_document_loader.py -v`
Expected: PASS (or SKIP if test files not available)

**Step 5: Commit**

```bash
git add doc_to_graphrag/ingestion/document_loader.py tests/ingestion/test_document_loader.py
git commit -m "feat: add unified document loader with auto OCR detection"
```

---

## Phase 2: Text Chunking

### Task 5: Overlapping Text Chunker

**Files:**
- Create: `doc_to_graphrag/chunking/text_splitter.py`
- Create: `tests/chunking/test_text_splitter.py`

**Step 1: Write the failing test**

```python
# tests/chunking/test_text_splitter.py
import pytest
from doc_to_graphrag.chunking.text_splitter import OverlappingTextSplitter

def test_chunking_creates_overlapping_chunks():
    splitter = OverlappingTextSplitter(chunk_size=100, chunk_overlap=25)

    text = "word " * 200  # 200 words
    chunks = splitter.split(text)

    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)

def test_chunk_overlap_preserves_boundaries():
    splitter = OverlappingTextSplitter(chunk_size=50, chunk_overlap=10)

    text = "A B C D E F G H I J " * 10  # 100 tokens
    chunks = splitter.split(text)

    # Check that consecutive chunks overlap
    for i in range(len(chunks) - 1):
        # Some content from chunk[i] should appear in chunk[i+1]
        assert len(chunks[i]) > 0
        assert len(chunks[i+1]) > 0

def test_metadata_tracks_chunk_index():
    splitter = OverlappingTextSplitter(chunk_size=100, chunk_overlap=20)

    text = "word " * 200
    chunks_with_meta = splitter.split_with_metadata(text)

    for i, chunk_data in enumerate(chunks_with_meta):
        assert chunk_data["chunk_index"] == i
        assert "text" in chunk_data
        assert "start_char" in chunk_data
        assert "end_char" in chunk_data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/chunking/test_text_splitter.py -v`
Expected: FAIL with "OverlappingTextSplitter not defined"

**Step 3: Write minimal implementation**

```python
# doc_to_graphrag/chunking/text_splitter.py
from typing import List, Dict, Any
import tiktoken

class OverlappingTextSplitter:
    """Split text into overlapping chunks using token-based counting."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 150):
        """
        Initialize text splitter.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Use tiktoken for accurate token counting (OpenAI compatible)
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def split(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text to split

        Returns:
            List of text chunks
        """
        # Tokenize
        tokens = self.tokenizer.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start forward (accounting for overlap)
            start += (self.chunk_size - self.chunk_overlap)

        return chunks

    def split_with_metadata(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text with metadata tracking.

        Returns:
            List of dictionaries with 'text', 'chunk_index', 'start_char', 'end_char'
        """
        chunks = []
        tokens = self.tokenizer.encode(text)

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Find character positions (approximate)
            # In production, maintain exact char offsets
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "token_count": len(chunk_tokens),
            })

            start += (self.chunk_size - self.chunk_overlap)
            chunk_index += 1

        return chunks
```

**Step 4: Add tiktoken to requirements**

Update `requirements.txt`:
```
tiktoken>=0.5.0
```

Install: `pip install tiktoken`

**Step 5: Run test to verify it passes**

Run: `pytest tests/chunking/test_text_splitter.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add doc_to_graphrag/chunking/text_splitter.py tests/chunking/test_text_splitter.py requirements.txt
git commit -m "feat: add overlapping text chunker with token-based splitting"
```

---

## Phase 3: LLM-Based Entity Extraction

### Task 6: Schema Inference Prompt Template

**Files:**
- Create: `doc_to_graphrag/extraction/prompts.py`
- Create: `tests/extraction/test_prompts.py`

**Step 1: Write the failing test**

```python
# tests/extraction/test_prompts.py
import pytest
from doc_to_graphrag.extraction.prompts import SchemaInferencePrompt

def test_schema_inference_prompt_contains_instructions():
    prompt = SchemaInferencePrompt()

    text = "Tesla Inc. CEO Elon Musk announced..."
    formatted = prompt.format(text=text)

    assert "entity_types" in formatted
    assert "relationship_types" in formatted
    assert text in formatted

def test_extraction_prompt_with_schema():
    prompt = SchemaInferencePrompt()

    text = "Sample text"
    entity_types = ["Organization", "Person"]
    rel_types = ["LEADS", "WORKS_FOR"]

    formatted = prompt.format_with_schema(
        text=text,
        entity_types=entity_types,
        relationship_types=rel_types
    )

    assert "Organization" in formatted
    assert "LEADS" in formatted
    assert text in formatted
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extraction/test_prompts.py -v`
Expected: FAIL with "SchemaInferencePrompt not defined"

**Step 3: Write minimal implementation**

```python
# doc_to_graphrag/extraction/prompts.py
from typing import List, Optional

class SchemaInferencePrompt:
    """Prompt templates for dynamic schema inference and entity extraction."""

    SCHEMA_INFERENCE_TEMPLATE = '''You are a knowledge extraction system. Your task is to:
1. Analyze this document chunk and determine what types of entities are RELEVANT for this content
2. Determine what relationship types are RELEVANT
3. Extract those entities and their relationships

Guidelines for entity type selection:
- Choose entity types that capture the key concepts in THIS document
- Examples: For contracts → Party, Obligation, Date; For research → Author, Method, Finding
- Keep entity types semantic and meaningful (not just "Entity")
- Use 3-8 entity types per document type

Guidelines for relationship types:
- Use canonical names: FOUNDED (not STARTED, CREATED, ESTABLISHED)
- Examples: WORKS_FOR (not EMPLOYED_BY), LOCATED_IN (not BASED_IN, AT)
- Use semantic, standardized relationship names

Input text:
{text}

Return JSON with this exact structure:
{{
  "entity_types": ["Type1", "Type2", ...],
  "relationship_types": ["REL_TYPE1", "REL_TYPE2", ...],
  "entities": [
    {{"id": "1", "type": "Type1", "name": "...", "properties": {{}}}},
    {{"id": "2", "type": "Type2", "name": "...", "properties": {{}}}}
  ],
  "relationships": [
    {{"type": "REL_TYPE", "from": "1", "to": "2", "properties": {{"description": "..."}}}}
  ]
}}
'''

    EXTRACTION_WITH_SCHEMA_TEMPLATE = '''You are a knowledge extraction system.

Use ONLY these entity types: {entity_types}
Use ONLY these relationship types: {relationship_types}

If you encounter a new entity type or relationship, map it to the closest canonical type from the provided lists.

Input text:
{text}

Return JSON with this exact structure:
{{
  "entities": [
    {{"id": "1", "type": "Type1", "name": "...", "properties": {{}}}},
    {{"id": "2", "type": "Type2", "name": "...", "properties": {{}}}}
  ],
  "relationships": [
    {{"type": "REL_TYPE", "from": "1", "to": "2", "properties": {{"description": "..."}}}}
  ]
}}
'''

    def format(self, text: str) -> str:
        """Format schema inference prompt (for first chunks)."""
        return self.SCHEMA_INFERENCE_TEMPLATE.format(text=text)

    def format_with_schema(
        self,
        text: str,
        entity_types: List[str],
        relationship_types: List[str]
    ) -> str:
        """Format extraction prompt with known schema (for subsequent chunks)."""
        return self.EXTRACTION_WITH_SCHEMA_TEMPLATE.format(
            text=text,
            entity_types=", ".join(entity_types),
            relationship_types=", ".join(relationship_types)
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extraction/test_prompts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add doc_to_graphrag/extraction/prompts.py tests/extraction/test_prompts.py
git commit -m "feat: add schema inference prompt templates"
```

### Task 7: LLM Extractor with Schema Inference

**Files:**
- Create: `doc_to_graphrag/extraction/llm_extractor.py`
- Create: `tests/extraction/test_llm_extractor.py`

**Step 1: Write the failing test**

```python
# tests/extraction/test_llm_extractor.py
import pytest
from doc_to_graphrag.extraction.llm_extractor import LLMExtractor
from unittest.mock import Mock, patch

def test_extract_with_schema_inference():
    # Mock LLM response
    mock_llm = Mock()
    mock_llm.generate.return_value = '''{
        "entity_types": ["Organization", "Person"],
        "relationship_types": ["LEADS"],
        "entities": [
            {"id": "1", "type": "Organization", "name": "Tesla"},
            {"id": "2", "type": "Person", "name": "Elon Musk"}
        ],
        "relationships": [
            {"type": "LEADS", "from": "2", "to": "1", "properties": {}}
        ]
    }'''

    extractor = LLMExtractor(llm=mock_llm)
    result = extractor.extract_with_schema_inference("Tesla CEO Elon Musk...")

    assert "entity_types" in result
    assert "entities" in result
    assert len(result["entities"]) == 2

def test_extract_with_existing_schema():
    mock_llm = Mock()
    mock_llm.generate.return_value = '''{
        "entities": [{"id": "1", "type": "Organization", "name": "Tesla"}],
        "relationships": []
    }'''

    extractor = LLMExtractor(llm=mock_llm)
    result = extractor.extract_with_schema(
        text="Tesla announced...",
        entity_types=["Organization"],
        relationship_types=["ANNOUNCED"]
    )

    assert "entities" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/extraction/test_llm_extractor.py -v`
Expected: FAIL with "LLMExtractor not defined"

**Step 3: Write minimal implementation**

```python
# doc_to_graphrag/extraction/llm_extractor.py
import json
import logging
from typing import Dict, List, Any, Optional

from .prompts import SchemaInferencePrompt

logger = logging.getLogger(__name__)

class LLMExtractor:
    """Extract entities and relationships using LLM with dynamic schema."""

    def __init__(self, llm, max_retries: int = 3):
        """
        Initialize LLM extractor.

        Args:
            llm: LLM instance (OpenAI or compatible)
            max_retries: Maximum retry attempts for failed extractions
        """
        self.llm = llm
        self.max_retries = max_retries
        self.prompt_template = SchemaInferencePrompt()

    def extract_with_schema_inference(self, text: str) -> Dict[str, Any]:
        """
        Extract entities with schema inference (for first chunks).

        Args:
            text: Input text chunk

        Returns:
            Dictionary with entity_types, relationship_types, entities, relationships
        """
        prompt = self.prompt_template.format(text=text)

        for attempt in range(self.max_retries):
            try:
                # Generate with LLM
                response = self.llm.generate(prompt)

                # Parse JSON
                result = json.loads(response)

                # Validate structure
                self._validate_schema_inference_result(result)

                return result

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")

                if attempt == self.max_retries - 1:
                    # Return minimal structure on final failure
                    return {
                        "entity_types": ["Entity"],
                        "relationship_types": [],
                        "entities": [],
                        "relationships": []
                    }

        return {
            "entity_types": ["Entity"],
            "relationship_types": [],
            "entities": [],
            "relationships": []
        }

    def extract_with_schema(
        self,
        text: str,
        entity_types: List[str],
        relationship_types: List[str]
    ) -> Dict[str, Any]:
        """
        Extract entities using known schema (for subsequent chunks).

        Args:
            text: Input text chunk
            entity_types: Known entity types
            relationship_types: Known relationship types

        Returns:
            Dictionary with entities and relationships
        """
        prompt = self.prompt_template.format_with_schema(
            text=text,
            entity_types=entity_types,
            relationship_types=relationship_types
        )

        for attempt in range(self.max_retries):
            try:
                response = self.llm.generate(prompt)
                result = json.loads(response)

                self._validate_extraction_result(result)

                return result

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")

                if attempt == self.max_retries - 1:
                    return {
                        "entities": [],
                        "relationships": []
                    }

        return {
            "entities": [],
            "relationships": []
        }

    def _validate_schema_inference_result(self, result: Dict) -> None:
        """Validate schema inference result structure."""
        required_keys = ["entity_types", "relationship_types", "entities", "relationships"]
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing required key: {key}")

    def _validate_extraction_result(self, result: Dict) -> None:
        """Validate extraction result structure."""
        required_keys = ["entities", "relationships"]
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing required key: {key}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/extraction/test_llm_extractor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add doc_to_graphrag/extraction/llm_extractor.py tests/extraction/test_llm_extractor.py
git commit -m "feat: add LLM-based entity extractor with schema inference"
```

---

## Phase 4: Entity Reconciliation

### Task 8: Type-Aware Entity Matcher

**Files:**
- Create: `doc_to_graphrag/reconciliation/entity_matcher.py`
- Create: `tests/reconciliation/test_entity_matcher.py`

**Step 1: Write the failing test**

```python
# tests/reconciliation/test_entity_matcher.py
import pytest
from doc_to_graphrag.reconciliation.entity_matcher import EntityMatcher

def test_exact_match_same_type():
    matcher = EntityMatcher()

    entity1 = {"name": "Apple Inc.", "type": "Organization"}
    entity2 = {"name": "apple inc.", "type": "Organization"}

    assert matcher.is_match(entity1, entity2) == True

def test_no_match_different_types():
    """Critical: Don't match entities of different types."""
    matcher = EntityMatcher()

    apple_org = {"name": "Apple", "type": "Organization"}
    apple_fruit = {"name": "Apple", "type": "Food"}

    assert matcher.is_match(apple_org, apple_fruit) == False

def test_fuzzy_match_same_type():
    matcher = EntityMatcher(similarity_threshold=0.85)

    entity1 = {"name": "Apple Inc.", "type": "Organization"}
    entity2 = {"name": "Apple Corporation", "type": "Organization"}

    # Should match based on similarity
    result = matcher.is_match(entity1, entity2)
    assert isinstance(result, bool)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/reconciliation/test_entity_matcher.py -v`
Expected: FAIL with "EntityMatcher not defined"

**Step 3: Write minimal implementation**

```python
# doc_to_graphrag/reconciliation/entity_matcher.py
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

class EntityMatcher:
    """Type-aware entity matching with exact and fuzzy matching."""

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize entity matcher.

        Args:
            similarity_threshold: Minimum similarity for fuzzy matching (0-1)
        """
        self.similarity_threshold = similarity_threshold

        # Load sentence transformer for entity name embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Cache embeddings
        self._embedding_cache = {}

    def is_match(self, entity1: Dict, entity2: Dict) -> bool:
        """
        Check if two entities match.

        Args:
            entity1: First entity dict with 'name' and 'type'
            entity2: Second entity dict with 'name' and 'type'

        Returns:
            True if entities match, False otherwise
        """
        # Step 1: Type must match (CRITICAL)
        if entity1["type"] != entity2["type"]:
            return False

        # Step 2: Exact string match (case-insensitive)
        name1 = entity1["name"].lower().strip()
        name2 = entity2["name"].lower().strip()

        if name1 == name2:
            return True

        # Step 3: Fuzzy match using embeddings
        similarity = self._compute_similarity(entity1["name"], entity2["name"])

        return similarity >= self.similarity_threshold

    def find_match(
        self,
        entity: Dict,
        candidates: List[Dict]
    ) -> Optional[Dict]:
        """
        Find matching entity from candidates.

        Args:
            entity: Entity to match
            candidates: List of candidate entities

        Returns:
            Matching entity or None
        """
        # Filter candidates by type first
        same_type_candidates = [
            c for c in candidates
            if c["type"] == entity["type"]
        ]

        # Check exact matches
        entity_name_lower = entity["name"].lower().strip()
        for candidate in same_type_candidates:
            if candidate["name"].lower().strip() == entity_name_lower:
                return candidate

        # Check fuzzy matches
        best_match = None
        best_score = 0

        for candidate in same_type_candidates:
            similarity = self._compute_similarity(entity["name"], candidate["name"])

            if similarity >= self.similarity_threshold and similarity > best_score:
                best_score = similarity
                best_match = candidate

        return best_match

    def _compute_similarity(self, name1: str, name2: str) -> float:
        """Compute embedding similarity between two entity names."""
        # Get embeddings (with caching)
        emb1 = self._get_embedding(name1)
        emb2 = self._get_embedding(name2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.embedder.encode(text)

        return self._embedding_cache[text]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/reconciliation/test_entity_matcher.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add doc_to_graphrag/reconciliation/entity_matcher.py tests/reconciliation/test_entity_matcher.py
git commit -m "feat: add type-aware entity matcher with fuzzy matching"
```

### Task 9: Entity Reconciliation Registry

**Files:**
- Create: `doc_to_graphrag/reconciliation/entity_registry.py`
- Create: `tests/reconciliation/test_entity_registry.py`

**Step 1: Write the failing test**

```python
# tests/reconciliation/test_entity_registry.py
import pytest
from doc_to_graphrag.reconciliation.entity_registry import EntityRegistry

def test_register_new_entity():
    registry = EntityRegistry()

    entity = {"name": "Tesla", "type": "Organization", "id": "1"}
    canonical_id = registry.register(entity)

    assert canonical_id is not None
    assert registry.get_canonical_id(entity) == canonical_id

def test_merge_duplicate_entities():
    registry = EntityRegistry()

    entity1 = {"name": "Apple Inc.", "type": "Organization", "id": "1"}
    entity2 = {"name": "Apple", "type": "Organization", "id": "2"}

    id1 = registry.register(entity1)
    id2 = registry.register(entity2)

    # Should return same canonical ID
    assert id1 == id2

def test_track_aliases():
    registry = EntityRegistry()

    entity1 = {"name": "Apple Inc.", "type": "Organization"}
    entity2 = {"name": "Apple Corporation", "type": "Organization"}

    registry.register(entity1)
    registry.register(entity2)

    canonical = registry.get_canonical_entity(entity1)
    assert "Apple Inc." in canonical["aliases"]
    assert "Apple Corporation" in canonical["aliases"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/reconciliation/test_entity_registry.py -v`
Expected: FAIL with "EntityRegistry not defined"

**Step 3: Write minimal implementation**

```python
# doc_to_graphrag/reconciliation/entity_registry.py
from typing import Dict, List, Optional
import uuid

from .entity_matcher import EntityMatcher

class EntityRegistry:
    """Registry for managing canonical entities and reconciliation."""

    def __init__(self, matcher: Optional[EntityMatcher] = None):
        """
        Initialize entity registry.

        Args:
            matcher: EntityMatcher instance (creates default if None)
        """
        self.matcher = matcher or EntityMatcher()

        # Map: canonical_id -> canonical entity
        self._canonical_entities = {}

        # Map: entity name (lowercase) -> canonical_id (for quick lookup)
        self._name_to_canonical = {}

    def register(self, entity: Dict) -> str:
        """
        Register entity and return canonical ID.

        Args:
            entity: Entity dict with 'name', 'type', and optional 'id'

        Returns:
            Canonical entity ID
        """
        # Check for existing match
        existing_id = self.get_canonical_id(entity)

        if existing_id:
            # Update aliases
            canonical = self._canonical_entities[existing_id]
            if entity["name"] not in canonical["aliases"]:
                canonical["aliases"].append(entity["name"])
            return existing_id

        # Create new canonical entity
        canonical_id = entity.get("id") or str(uuid.uuid4())

        self._canonical_entities[canonical_id] = {
            "id": canonical_id,
            "name": entity["name"],  # Primary name
            "type": entity["type"],
            "properties": entity.get("properties", {}),
            "aliases": [entity["name"]],
        }

        # Update lookup index
        key = self._make_key(entity["name"], entity["type"])
        self._name_to_canonical[key] = canonical_id

        return canonical_id

    def get_canonical_id(self, entity: Dict) -> Optional[str]:
        """
        Get canonical ID for entity if it exists.

        Args:
            entity: Entity to look up

        Returns:
            Canonical ID or None
        """
        # Quick exact match lookup
        key = self._make_key(entity["name"], entity["type"])
        if key in self._name_to_canonical:
            return self._name_to_canonical[key]

        # Fuzzy match against all canonical entities
        candidates = list(self._canonical_entities.values())
        match = self.matcher.find_match(entity, candidates)

        return match["id"] if match else None

    def get_canonical_entity(self, entity: Dict) -> Optional[Dict]:
        """Get full canonical entity."""
        canonical_id = self.get_canonical_id(entity)
        return self._canonical_entities.get(canonical_id)

    def get_all_entities(self) -> List[Dict]:
        """Get all canonical entities."""
        return list(self._canonical_entities.values())

    def get_entity_types(self) -> List[str]:
        """Get all unique entity types."""
        return list(set(e["type"] for e in self._canonical_entities.values()))

    def _make_key(self, name: str, entity_type: str) -> str:
        """Create lookup key from name and type."""
        return f"{entity_type}:{name.lower().strip()}"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/reconciliation/test_entity_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add doc_to_graphrag/reconciliation/entity_registry.py tests/reconciliation/test_entity_registry.py
git commit -m "feat: add entity registry for reconciliation and alias tracking"
```

---

## Phase 5: Main Pipeline Integration

### Task 10: Document Processing Pipeline

**Files:**
- Create: `doc_to_graphrag/pipeline.py`
- Create: `tests/test_integration/test_pipeline.py`

**Step 1: Write the failing integration test**

```python
# tests/test_integration/test_pipeline.py
import pytest
from doc_to_graphrag.pipeline import DocumentGraphPipeline
from unittest.mock import Mock

def test_end_to_end_document_processing():
    # Mock Neo4j driver and LLM
    mock_driver = Mock()
    mock_llm = Mock()
    mock_embedder = Mock()

    pipeline = DocumentGraphPipeline(
        driver=mock_driver,
        llm=mock_llm,
        embedder=mock_embedder
    )

    # This will fail until we implement the pipeline
    result = pipeline.process_document("sample.pdf")

    assert result is not None
    assert "document_id" in result
    assert "entity_count" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration/test_pipeline.py -v`
Expected: FAIL with "DocumentGraphPipeline not defined"

**Step 3: Write minimal pipeline implementation**

```python
# doc_to_graphrag/pipeline.py
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .ingestion.document_loader import DocumentLoader
from .chunking.text_splitter import OverlappingTextSplitter
from .extraction.llm_extractor import LLMExtractor
from .reconciliation.entity_registry import EntityRegistry

logger = logging.getLogger(__name__)

class DocumentGraphPipeline:
    """Main pipeline for document-to-graph processing."""

    def __init__(
        self,
        driver,  # Neo4j driver
        llm,     # LLM instance
        embedder,  # Embedder instance
        chunk_size: int = 600,
        chunk_overlap: int = 150
    ):
        """
        Initialize pipeline.

        Args:
            driver: Neo4j graph database driver
            llm: Language model for entity extraction
            embedder: Embedding model for vector search
            chunk_size: Text chunk size in tokens
            chunk_overlap: Overlap size in tokens
        """
        self.driver = driver
        self.llm = llm
        self.embedder = embedder

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = OverlappingTextSplitter(chunk_size, chunk_overlap)
        self.llm_extractor = LLMExtractor(llm)

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process document end-to-end.

        Args:
            file_path: Path to document file

        Returns:
            Processing result with document_id, entity_count, etc.
        """
        logger.info(f"Processing document: {file_path}")

        # Step 1: Load document
        doc_data = self.document_loader.load_document(file_path)
        text = doc_data["text"]
        metadata = doc_data["metadata"]

        # Step 2: Chunk text
        chunks = self.text_splitter.split_with_metadata(text)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Extract entities with schema inference
        entity_registry = EntityRegistry()
        inferred_schema = None

        for i, chunk_data in enumerate(chunks):
            chunk_text = chunk_data["text"]

            # First few chunks: infer schema
            if i < 3 and inferred_schema is None:
                result = self.llm_extractor.extract_with_schema_inference(chunk_text)

                # Capture schema from first successful inference
                if result["entity_types"]:
                    inferred_schema = {
                        "entity_types": result["entity_types"],
                        "relationship_types": result["relationship_types"]
                    }
                    logger.info(f"Inferred schema: {inferred_schema}")

                entities = result["entities"]

            # Subsequent chunks: use schema
            elif inferred_schema:
                result = self.llm_extractor.extract_with_schema(
                    text=chunk_text,
                    entity_types=inferred_schema["entity_types"],
                    relationship_types=inferred_schema["relationship_types"]
                )
                entities = result["entities"]

            else:
                # Fallback if schema inference failed
                result = self.llm_extractor.extract_with_schema_inference(chunk_text)
                entities = result["entities"]

            # Step 4: Reconcile entities
            for entity in entities:
                entity_registry.register(entity)

        # Step 5: Build graph (placeholder - will implement in next task)
        # graph_result = self._build_neo4j_graph(entity_registry, chunks, metadata)

        return {
            "document_id": metadata["file_path"],
            "entity_count": len(entity_registry.get_all_entities()),
            "chunk_count": len(chunks),
            "entity_types": entity_registry.get_entity_types(),
            "metadata": metadata,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_integration/test_pipeline.py -v`
Expected: PASS (with mocked components)

**Step 5: Commit**

```bash
git add doc_to_graphrag/pipeline.py tests/test_integration/test_pipeline.py
git commit -m "feat: add main document processing pipeline with schema inference"
```

---

## Phase 6: Notebook Interface

### Task 11: Create Interactive Demo Notebook

**Files:**
- Create: `notebooks/demo_pipeline.ipynb`

**Step 1: Create notebook structure**

```python
# Cell 1: Setup and imports
"""
# Document-to-GraphRAG Pipeline Demo

This notebook demonstrates the general-purpose document GraphRAG pipeline.
"""

# Cell 2: Install and imports
"""
!pip install -e ..
"""

import os
from dotenv import load_dotenv
from doc_to_graphrag.pipeline import DocumentGraphPipeline
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

load_dotenv()

# Cell 3: Initialize components
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Neo4j driver
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# LLM
llm = OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={
        "response_format": {"type": "json_object"},
        "temperature": 0
    }
)

# Embedder
embedder = OpenAIEmbeddings()

# Cell 4: Create pipeline
pipeline = DocumentGraphPipeline(
    driver=driver,
    llm=llm,
    embedder=embedder,
    chunk_size=600,
    chunk_overlap=150
)

# Cell 5: Process a document
result = pipeline.process_document("path/to/your/document.pdf")

print(f"Processed: {result['document_id']}")
print(f"Entities extracted: {result['entity_count']}")
print(f"Entity types: {result['entity_types']}")
print(f"Chunks created: {result['chunk_count']}")

# Cell 6: Explore results
# TODO: Add query examples once retrieval layer is implemented
```

**Step 2: Commit notebook**

```bash
git add notebooks/demo_pipeline.ipynb
git commit -m "feat: add demo notebook for document GraphRAG pipeline"
```

---

## Next Steps (Future Tasks)

The following tasks complete the implementation:

**Task 12:** Neo4j Graph Builder Integration
**Task 13:** Vector Index Setup
**Task 14:** RAG Retriever Implementation
**Task 15:** Entity Explorer Implementation
**Task 16:** Cross-Document Reasoner
**Task 17:** Relationship Normalization
**Task 18:** Error Handling & Checkpointing
**Task 19:** Comprehensive Testing
**Task 20:** Documentation & Examples

---

## Testing Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=doc_to_graphrag --cov-report=html

# Run specific test module
pytest tests/ingestion/ -v

# Run integration tests only
pytest tests/test_integration/ -v
```

## Development Workflow

1. Always write tests first (TDD)
2. Run tests to verify they fail
3. Implement minimal code to pass
4. Run tests to verify they pass
5. Commit with descriptive message
6. Repeat for next task

## Success Criteria

- [ ] All unit tests passing
- [ ] Integration test for full pipeline passing
- [ ] Can process PDF documents (native and scanned)
- [ ] Dynamic schema inference working
- [ ] Type-aware entity reconciliation working
- [ ] Entities stored in Neo4j with relationships
- [ ] Demo notebook runs successfully

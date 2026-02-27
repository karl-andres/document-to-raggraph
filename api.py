"""FastAPI server for the Document-to-GraphRAG pipeline."""

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pipeline import GraphRagPipeline

# ── Logging ───────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Pydantic models ──────────────────────────────────────────────────── #


class DocumentResponse(BaseModel):
    file_name: str
    status: Literal["success", "error"]
    node_labels: Optional[List[str]] = None
    rel_types: Optional[List[str]] = None
    kg_result: Optional[str] = None
    error: Optional[str] = None


class BatchSubmitResponse(BaseModel):
    batch_id: str
    total: int
    message: str = "Batch submitted. Poll /batch/{batch_id}/status for progress."


class BatchStatusResponse(BaseModel):
    batch_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    total: int
    completed: int
    succeeded: int
    failed: int
    results: List[DocumentResponse] = Field(default_factory=list)


# ── App + pipeline ───────────────────────────────────────────────────── #

app = FastAPI(
    title="Document-to-GraphRAG API",
    description="Upload documents and build a Neo4j knowledge graph automatically.",
    version="1.0.0",
)

# Initialize the pipeline once at module load (loads LLM, Neo4j, etc.)
pipeline = GraphRagPipeline()

# In-memory batch tracker
_batches: Dict[str, BatchStatusResponse] = {}

# Temp directory for file uploads
_upload_dir = Path(tempfile.mkdtemp(prefix="graphrag_uploads_"))


# ── Helpers ───────────────────────────────────────────────────────────── #


def _save_upload(upload: UploadFile) -> str:
    """Save an uploaded file to the temp directory; return its path."""
    dest = _upload_dir / f"{uuid.uuid4().hex}_{upload.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest)


def _to_response(result: dict, file_name: str) -> DocumentResponse:
    schema = result.get("schema") or {}
    return DocumentResponse(
        file_name=file_name,
        status=result["status"],
        node_labels=schema.get("node_labels"),
        rel_types=schema.get("rel_types"),
        kg_result=result.get("kg_result"),
        error=result.get("error"),
    )


# ── Endpoints ─────────────────────────────────────────────────────────── #


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/process", response_model=DocumentResponse)
async def process_single(
    file: UploadFile = File(...),
    force_ocr: bool = Query(False, description="Force OCR even on native-text PDFs"),
):
    """Process a single uploaded document through the full pipeline."""
    saved_path = _save_upload(file)
    try:
        result = await pipeline.scan_single_document(saved_path, force_ocr=force_ocr)
        return _to_response(result, file.filename)
    finally:
        try:
            os.remove(saved_path)
        except OSError:
            pass


@app.post("/batch", response_model=BatchSubmitResponse)
async def submit_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    force_ocr: bool = Query(False, description="Force OCR on all documents"),
    batch_size: int = Query(5, ge=1, le=20, description="Max concurrent documents"),
):
    """
    Submit multiple documents for background processing.
    Returns a batch_id immediately — poll /batch/{batch_id}/status for progress.
    """
    batch_id = uuid.uuid4().hex[:12]

    # Save all files, remember original names
    saved: List[tuple] = [(  _save_upload(f), f.filename) for f in files]

    _batches[batch_id] = BatchStatusResponse(
        batch_id=batch_id,
        status="pending",
        total=len(saved),
        completed=0,
        succeeded=0,
        failed=0,
    )

    background_tasks.add_task(_run_batch, batch_id, saved, force_ocr, batch_size)
    return BatchSubmitResponse(batch_id=batch_id, total=len(saved))


@app.get("/batch/{batch_id}/status", response_model=BatchStatusResponse)
async def batch_status(batch_id: str):
    """Poll the status of a submitted batch."""
    if batch_id not in _batches:
        return JSONResponse(status_code=404, content={"detail": f"Batch {batch_id} not found."})
    return _batches[batch_id]


# ── Background batch runner ──────────────────────────────────────────── #


async def _run_batch(
    batch_id: str,
    saved_files: List[tuple],
    force_ocr: bool,
    batch_size: int,
):
    batch = _batches[batch_id]
    batch.status = "processing"

    file_paths = [path for path, _ in saved_files]
    name_map = {path: name for path, name in saved_files}

    try:
        results = await pipeline.scan_batch(
            file_paths, force_ocr=force_ocr, batch_size=batch_size
        )
        for r in results:
            resp = _to_response(r, name_map.get(r["file_path"], "unknown"))
            batch.results.append(resp)
            batch.completed += 1
            if resp.status == "success":
                batch.succeeded += 1
            else:
                batch.failed += 1

        batch.status = "completed" if batch.failed == 0 else "failed"
    except Exception as exc:
        logger.exception(f"Batch {batch_id} failed")
        batch.status = "failed"
    finally:
        for path, _ in saved_files:
            try:
                os.remove(path)
            except OSError:
                pass


# ── Run directly ─────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

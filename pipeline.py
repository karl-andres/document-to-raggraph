"""GraphRAG Pipeline — loads models once, exposes scan_single_document & scan_batch."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import neo4j
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

from doc_to_graphrag.extraction import SchemaInferrer
from doc_to_graphrag.ingestion import DocumentLoader

logger = logging.getLogger(__name__)


# ── Prompt template ──────────────────────────────────────────────────── #

PROMPT_TEMPLATE = """\
You are a Knowledge Engineer task with extracting structured information from unstructured text \
to build a comprehensive property graph for advanced data analysis and retrieval.

Extract the entities (nodes) and specify their type from the following Input text based on the provided schema. \
Also, extract the directed relationships between these nodes.

Return the result strictly as a JSON object using the following format:
{{"nodes": [ {{"id": "unique_id", "label": "Entity_Type", "properties": {{"name": "Entity Name", "description": "Entity Description", "source_text": "Entity Source Text" }} }}],
  "relationships": [{{"type": "RELATIONSHIP_TYPE", "start_node_id": "unique_id", "end_node_id": "unique_id", "properties": {{"details": "Brief description of how they interact"}} }}] }}

---

### Constraints:
1. **Schema Adherence:** Use only the following node labels and relationship types:
{schema}

2. **Node IDs:** Assign a unique string ID to each node and use these IDs to define the relationships.
3. **Directionality:** Ensure the relationship direction (start_node to end_node) reflects the logic of the source text.
4. **Format:** Do not include any conversational text, preamble, or markdown formatting outside of the JSON block.

---

### Examples:
{examples}

---

### Input text:
{text}
"""


class GraphRagPipeline:
    """
    Initializes all heavy resources (Neo4j driver, LLM, embedder, text splitter)
    once and exposes two methods:

        scan_single_document(file_path)  →  dict
        scan_batch(file_paths)           →  list[dict]
    """

    def __init__(self):
        load_dotenv()

        # ── Neo4j ─────────────────────────────────────────────────────── #
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USERNAME")
        neo4j_pass = os.getenv("NEO4J_PASSWORD")

        if not all([neo4j_uri, neo4j_user, neo4j_pass]):
            raise RuntimeError(
                "Missing Neo4j credentials. "
                "Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env"
            )

        self.neo4j_driver = neo4j.GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_pass)
        )
        logger.info(f"Connected to Neo4j at {neo4j_uri}")

        # ── LLM + Embedder ────────────────────────────────────────────── #
        self.llm = OpenAILLM(
            model_name="gpt-4o-mini",
            model_params={
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
        )
        self.embedder = OpenAIEmbeddings()

        # ── Text splitter ─────────────────────────────────────────────── #
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            length_function=len,
            separators=[
                "\n\n", "\n", " ", ".", ",",
                "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", "",
            ],
            is_separator_regex=False,
        )
        self.adapted_splitter = LangChainTextSplitterAdapter(self.text_splitter)

        # ── Reusable components ───────────────────────────────────────── #
        self.loader = DocumentLoader()
        self.inferrer = SchemaInferrer(
            self.llm, initial_chunk_count=3, refine_sample_count=5
        )

        self.prompt_template = PROMPT_TEMPLATE

    # ------------------------------------------------------------------ #
    #  Single document                                                     #
    # ------------------------------------------------------------------ #

    async def scan_single_document(
        self, file_path: str, force_ocr: bool = False
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on one document:
        load → chunk → infer schema → extract KG → write to Neo4j.

        Returns:
            {
                "file_path": str,
                "status": "success" | "error",
                "schema": {"node_labels": [...], "rel_types": [...]},
                "kg_result": <SimpleKGPipeline result> | None,
                "error": str | None,
            }
        """
        try:
            # 1. Load document
            logger.info(f"Loading: {file_path}")
            doc = self.loader.load(file_path, force_ocr=force_ocr)
            extracted_text = doc["text"]

            # 2. Chunk
            chunks = self.text_splitter.split_text(extracted_text)
            logger.info(f"  → {len(chunks)} chunks")

            # 3. Infer schema
            schema = self.inferrer.infer(chunks)
            logger.info(
                f"  → Schema: {len(schema['node_labels'])} labels, "
                f"{len(schema['rel_types'])} rels"
            )

            # 4. Build KG via SimpleKGPipeline
            kg_builder = SimpleKGPipeline(
                llm=self.llm,
                driver=self.neo4j_driver,
                embedder=self.embedder,
                entities=schema["node_labels"],
                relations=schema["rel_types"],
                from_pdf=False,
                text_splitter=self.adapted_splitter,
                prompt_template=self.prompt_template,
            )
            kg_result = await kg_builder.run_async(text=extracted_text)
            logger.info(f"  ✓ Done: {file_path}")

            return {
                "file_path": file_path,
                "status": "success",
                "schema": schema,
                "kg_result": str(kg_result),
                "error": None,
            }

        except Exception as exc:
            logger.exception(f"Failed to process {file_path}")
            return {
                "file_path": file_path,
                "status": "error",
                "schema": None,
                "kg_result": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------ #
    #  Batch processing                                                    #
    # ------------------------------------------------------------------ #

    async def scan_batch(
        self,
        file_paths: List[str],
        force_ocr: bool = False,
        batch_size: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents concurrently.

        Uses asyncio.Semaphore to limit concurrency so the LLM API
        and Neo4j aren't overwhelmed.

        Args:
            file_paths:  List of document paths.
            force_ocr:   Force OCR on all documents.
            batch_size:  Max documents processed in parallel (default 5).

        Returns:
            List of result dicts (same shape as scan_single_document).
        """
        semaphore = asyncio.Semaphore(batch_size)

        async def _limited(path: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.scan_single_document(path, force_ocr=force_ocr)

        logger.info(
            f"Batch start: {len(file_paths)} documents, concurrency={batch_size}"
        )
        results = await asyncio.gather(*[_limited(p) for p in file_paths])

        succeeded = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        logger.info(f"Batch done: {succeeded} succeeded, {failed} failed")

        return list(results)
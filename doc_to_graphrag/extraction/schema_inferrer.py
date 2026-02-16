"""LLM-based schema inference — discovers entity types and relationship types from text."""

import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────── #

SCHEMA_INFERENCE_PROMPT = """\
You are a Knowledge Engineer. Analyze the following text chunks and determine:

1. **Node labels (entity types)** — the categories of things mentioned in this text.
   - Choose types that are meaningful (not generic like "Entity" or "Thing").
   - Examples by domain: Contracts → Party, Obligation, Clause, Date;
     Research → Author, Method, Finding, Dataset;
     Business → Organization, Person, Product, Location.
   - Return 3–10 types.

2. **Relationship types** — the ways these entities relate to each other.
   - Use UPPER_SNAKE_CASE.
   - Prefer canonical names: FOUNDED (not STARTED/CREATED), WORKS_FOR (not EMPLOYED_BY), \
LOCATED_IN (not BASED_IN).
   - Return 3–15 types.

Return a JSON object with exactly these keys:
{
  "node_labels": ["Label1", "Label2", ...],
  "rel_types": ["REL_TYPE_1", "REL_TYPE_2", ...],
  "reasoning": "Brief explanation of why you chose these types"
}
"""

SCHEMA_REFINEMENT_PROMPT = """\
You are a Knowledge Engineer. You previously inferred these entity types and relationship \
types from the first chunks of a document:

Current node labels: {node_labels}
Current relationship types: {rel_types}

Now review the following additional text chunk. If needed, suggest NEW entity types or \
relationship types that are missing. Only add types that are clearly needed — do not \
duplicate or create near-synonyms of existing types.

Return a JSON object:
{{
  "new_node_labels": ["NewLabel1", ...],
  "new_rel_types": ["NEW_REL_TYPE", ...],
  "reasoning": "Brief explanation (or 'No new types needed')"
}}

If no new types are needed, return empty lists.
"""


class SchemaInferrer:
    """
    Infer a graph schema (node labels + relationship types) from document chunks
    using an LLM.

    Two-step process:
      1. Base inference — send the first N chunks to discover initial types.
      2. Refinement — scan remaining chunks for new types not yet captured.

    Usage:
        inferrer = SchemaInferrer(llm)
        schema = inferrer.infer(chunks)
        print(schema["node_labels"])  # ["Organization", "Person", ...]
        print(schema["rel_types"])    # ["FOUNDED", "WORKS_FOR", ...]
    """

    def __init__(self, llm, initial_chunk_count: int = 3, refine_sample_count: int = 5):
        """
        Args:
            llm: Any object with an `invoke(input, system_instruction)` method
                 that returns an object with a `.content` string (e.g. neo4j_graphrag OpenAILLM).
            initial_chunk_count: How many chunks to send for base schema inference.
            refine_sample_count: How many additional chunks to sample for refinement.
        """
        self.llm = llm
        self.initial_chunk_count = initial_chunk_count
        self.refine_sample_count = refine_sample_count

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def infer(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Infer node labels and relationship types from a list of text chunks.

        Args:
            chunks: Ordered list of text chunks from the document.

        Returns:
            {
                "node_labels": ["Label1", "Label2", ...],
                "rel_types": ["REL_TYPE_1", ...],
            }
        """
        # Step 1: Base inference from the first N chunks
        base_chunks = chunks[: self.initial_chunk_count]
        combined_text = "\n\n---\n\n".join(base_chunks)

        logger.info(f"Step 1: Inferring schema from first {len(base_chunks)} chunks...")
        base_schema = self._call_llm_for_schema(combined_text)

        node_labels = list(set(base_schema.get("node_labels", [])))
        rel_types = list(set(base_schema.get("rel_types", [])))

        logger.info(f"  → Base node labels: {node_labels}")
        logger.info(f"  → Base rel types:   {rel_types}")

        # Step 2: Refine by sampling later chunks
        remaining = chunks[self.initial_chunk_count :]
        if remaining and self.refine_sample_count > 0:
            # Evenly sample from the remaining chunks
            step = max(1, len(remaining) // self.refine_sample_count)
            sample_chunks = remaining[::step][: self.refine_sample_count]

            logger.info(f"Step 2: Refining schema with {len(sample_chunks)} additional sample chunks...")
            for chunk in sample_chunks:
                new_types = self._call_llm_for_refinement(chunk, node_labels, rel_types)
                added_labels = new_types.get("new_node_labels", [])
                added_rels = new_types.get("new_rel_types", [])

                for label in added_labels:
                    if label not in node_labels:
                        node_labels.append(label)
                        logger.info(f"  + Added node label: {label}")
                for rel in added_rels:
                    if rel not in rel_types:
                        rel_types.append(rel)
                        logger.info(f"  + Added rel type: {rel}")

        logger.info(f"Final schema — {len(node_labels)} node labels, {len(rel_types)} rel types")
        return {"node_labels": node_labels, "rel_types": rel_types}

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _call_llm_for_schema(self, text: str) -> Dict[str, Any]:
        """Send the base inference prompt to the LLM."""
        response = self.llm.invoke(
            input=text,
            system_instruction=SCHEMA_INFERENCE_PROMPT,
        )
        return self._parse_json(response.content)

    def _call_llm_for_refinement(
        self, text: str, node_labels: List[str], rel_types: List[str]
    ) -> Dict[str, Any]:
        """Send the refinement prompt to the LLM."""
        system = SCHEMA_REFINEMENT_PROMPT.format(
            node_labels=json.dumps(node_labels),
            rel_types=json.dumps(rel_types),
        )
        response = self.llm.invoke(input=text, system_instruction=system)
        return self._parse_json(response.content)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        cleaned = text.strip()
        # Strip ```json ... ``` wrappers
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}\nRaw text: {text[:300]}")
            return {}

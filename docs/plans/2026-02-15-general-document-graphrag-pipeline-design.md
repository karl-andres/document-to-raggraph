# General-Purpose Document-to-Graph Knowledge Extraction Pipeline

**Date:** 2026-02-15
**Status:** Design Approved
**Author:** Design Session with User

## Overview

A reusable pipeline that transforms unstructured documents (PDFs, text files, scanned documents) into a structured Neo4j knowledge graph that supports RAG, entity exploration, cross-document reasoning, and semantic search.

### Key Requirements

- **Document Types:** Mixed/general documents (contracts, reports, emails, articles)
- **Use Cases:** RAG Q&A, entity-centric exploration, cross-document reasoning, document discovery (all equally important)
- **Scale:** Start small but architect for future scale
- **Graph Database:** Neo4j (leverage existing knowledge)
- **OCR:** Critical - must handle scanned documents well
- **Interface:** Jupyter notebook (interactive experimentation)

## Architecture

### Pipeline Flow

```
Document Input (PDF/scan)
    ↓
1. Document Ingestion & OCR
    ↓
2. Text Chunking (overlapping chunks)
    ↓
3. Per-Chunk LLM Processing
   - Infer relevant entity types for this document
   - Extract entities and relationships
    ↓
4. Entity Reconciliation (type-aware)
   - Merge duplicate entities across chunks
   - Normalize relationship types
    ↓
5. Graph Construction (SimpleKGPipeline)
   - Create nodes: Document, Chunk, Entity
   - Create edges: FROM_CHUNK, entity relationships
   - Generate embeddings
    ↓
6. Query Layer
   - Vector search (RAG)
   - Cypher queries (entity exploration)
   - Hybrid retrieval (graph + vector)
```

### Core Design Principles

1. **Modularity:** Each stage can be swapped/improved independently
2. **Dynamic Schema:** LLM infers entity types per document rather than hardcoding
3. **Type-Aware Processing:** Entity reconciliation and relationship normalization respect entity types
4. **Source Attribution:** Every entity tracks which chunks mention it via `FROM_CHUNK` relationships

## Component Design

### 1. Document Ingestion & OCR

**Supported Formats:**
- Native PDFs (text-based) → `pdfplumber`
- Scanned PDFs/Images → OCR pipeline
- Text files → Direct ingestion

**OCR Pipeline:**
```python
if is_scanned_pdf_or_image(file):
    # 1. Image preprocessing
    images = convert_to_images(file)
    preprocessed = enhance_images(images)  # Deskew, denoise, contrast

    # 2. OCR extraction
    text = ocr_engine.extract(preprocessed)  # pytesseract initially

    # 3. Confidence filtering
    text = filter_low_confidence_text(text)
else:
    text = pdfplumber.extract_text(file)
```

**OCR Technology:**
- **Start:** `pytesseract` (local, free, good for English)
- **Upgrade Path:** Google Cloud Vision or AWS Textract (better accuracy, multi-language)

**Document Metadata:**
Each `Document` node stores:
- `file_path`, `file_type`, `ocr_used` (boolean)
- `created_at`, `page_count`
- `ocr_confidence_avg` (if OCR used)

### 2. Chunking Strategy

**Configuration:**
```python
FixedSizeSplitter(
    chunk_size=600,      # tokens
    chunk_overlap=150    # 25% overlap
)
```

**Why These Numbers:**
- **600 tokens:** Enough context for LLM without overwhelming
- **150 token overlap:** Prevents splitting entities at boundaries

**Chunk-Entity Relationship:**
```
(Document) -[:HAS_CHUNK]-> (Chunk)
(Chunk) -[:FROM_CHUNK]-> (Entity)
(Entity) -[:RELATES_TO]-> (Entity)
```

**Example:**
- Chunk 1 mentions "Apple Inc."
- Chunk 2 mentions "Apple" and "iPhone"
- Entity reconciliation merges both "Apple" references
- Final: `(Apple)-[:FROM_CHUNK]->(Chunk1)`, `(Apple)-[:FROM_CHUNK]->(Chunk2)`

**RAG Benefit:** When answering "Who founded Apple?", retrieve relevant chunks AND traverse graph to find connected entities with source attribution.

### 3. LLM-Based Entity Extraction with Dynamic Schema

**The Core Innovation:** LLM infers entity types per document rather than using predefined types.

**Prompt Template:**
```python
prompt_template = '''
You are a knowledge extraction system. Your task is to:
1. Analyze this document chunk and determine what types of entities are RELEVANT
2. Extract those entities and their relationships
3. Use canonical relationship names

Guidelines for entity type selection:
- Choose types that capture key concepts in THIS document
- Examples: Contracts → Party, Obligation, Date; Research → Author, Method, Finding
- Keep types semantic and meaningful (not just "Entity")
- Use 3-8 entity types per document type

Guidelines for relationship types:
- Use canonical names: FOUNDED (not STARTED, CREATED)
- WORKS_FOR (not EMPLOYED_BY, HIRED_BY)
- LOCATED_IN (not BASED_IN, AT)

Input text:
{text}

Return JSON:
{{
  "entity_types": ["Type1", "Type2", ...],
  "relationship_types": ["REL_TYPE1", "REL_TYPE2", ...],
  "entities": [
    {{"id": "1", "type": "Type1", "name": "...", "properties": {{}}}},
    ...
  ],
  "relationships": [
    {{"type": "REL_TYPE", "from": "1", "to": "2", "properties": {{"description": "..."}}}}
  ]
}}
'''
```

**Two-Step Process:**

**Step 1: Schema Inference (first 2-3 chunks)**
- LLM returns `entity_types` and `relationship_types` it determined are relevant
- Store these as the document's schema

**Step 2: Consistent Extraction (remaining chunks)**
- Pass inferred schema to LLM: "Use these entity types: {entity_types}"
- Ensures consistency across chunks

**Example Output:**
```json
{
  "entity_types": ["Organization", "Person", "Location", "Event"],
  "relationship_types": ["LEADS", "ANNOUNCED", "LOCATED_IN"],
  "entities": [
    {"id": "1", "type": "Organization", "name": "Tesla Inc."},
    {"id": "2", "type": "Person", "name": "Elon Musk", "properties": {"role": "CEO"}},
    {"id": "3", "type": "Location", "name": "Berlin"}
  ],
  "relationships": [
    {"type": "LEADS", "from": "2", "to": "1"}
  ]
}
```

### 4. Entity Reconciliation (Type-Aware)

**The Challenge:** Same entity appears with variations across chunks:
- "Apple Inc.", "Apple", "Apple Corporation"

**Critical Constraint:** Only reconcile entities of the **same type** to avoid:
- "Apple" (Organization) merging with "Apple" (Fruit)

**Three-Stage Matching:**

```python
def find_matching_entity(entity, entity_index, similarity_threshold=0.85):
    # Step 1: Filter candidates by entity type (REQUIRED)
    candidates = [e for e in entity_index.values() if e.type == entity.type]

    # Step 2: Exact match within type
    for candidate in candidates:
        if entity.name.lower() == candidate.name.lower():
            return candidate.id

    # Step 3: Fuzzy match within type (embedding similarity)
    for candidate in candidates:
        similarity = embedding_similarity(entity.name, candidate.name)
        if similarity > similarity_threshold:
            return candidate.id

    # Step 4: LLM disambiguation (if multiple matches)
    if len(potential_matches) > 1:
        return llm_disambiguate(entity, potential_matches)

    return None  # New entity
```

**Matching Hierarchy:**
1. **Type match** (required): Organization vs Organization, Person vs Person
2. **Exact string match** (within type)
3. **Fuzzy embedding match** (within type, threshold > 0.85)
4. **LLM disambiguation** (within type, if needed)

**Neo4j Schema After Reconciliation:**
```
(Entity {
  name: "Apple Inc.",
  type: "Organization",
  aliases: ["Apple", "Apple Corp"]
})
  ├─[:FROM_CHUNK]→ (Chunk1)
  ├─[:FROM_CHUNK]→ (Chunk2)
  └─[:FROM_CHUNK]→ (Chunk5)
```

### 5. Relationship Normalization

**Problem:** Same relationship expressed differently:
- `(Elon)-[:FOUNDED]->(Tesla)`
- `(Elon)-[:STARTED]->(Tesla)`

**Solution:** Include relationships in schema inference (Step 1) to establish canonical types.

**Relationship Registry:**
```python
relationship_registry = {
    "FOUNDED": ["started", "created", "established"],
    "WORKS_FOR": ["employed by", "hired by"],
    "LOCATED_IN": ["based in", "at"]
}
```

**In Neo4j:**
```
(Elon)-[:FOUNDED {
  mentioned_as: ["founded", "started"],
  confidence: 0.95
}]->(Tesla)
```

### 6. Graph Construction with Neo4j

**Node Types:**
```
(Document {file_path, file_type, ocr_used, created_at})
(Chunk {text, chunk_id, chunk_index, embedding})
(Entity {name, type, properties, aliases})
```

**Relationship Types:**
```
(Document)-[:HAS_CHUNK]->(Chunk)
(Entity)-[:FROM_CHUNK]->(Chunk)
(Entity)-[:RELATES_TO {type, description}]->(Entity)
```

**Integration with SimpleKGPipeline:**

```python
# Pre-processing
chunks = chunk_document(ocr_text)
entity_registry = {}

for chunk in chunks:
    result = llm_extract_with_schema_inference(chunk)
    reconciled = reconcile_entities(result.entities, entity_registry)

# SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    embedder=embedder,
    entities=get_all_entity_types(entity_registry),  # Dynamic
    relations=get_all_relation_types(),  # Dynamic
    from_pdf=False  # Already extracted
)

# Load into Neo4j
for chunk, entities in zip(chunks, entities_per_chunk):
    kg_builder.add_chunk(chunk, entities)
```

**Vector Index:**
```python
create_vector_index(
    driver,
    name="chunk_embeddings",
    label="Chunk",
    embedding_property="embedding",
    dimensions=1536,
    similarity_fn="cosine"
)
```

### 7. Query & Retrieval Layer

**Four Query Modes (All Equally Supported):**

#### 1. RAG Question-Answering

**Pattern:** Vector search + Graph enrichment

```python
def rag_query(question: str, top_k: int = 5):
    # Vector search for relevant chunks
    similar_chunks = vector_retriever.search(question, top_k=top_k)

    # Enrich with graph context
    context = []
    for chunk in similar_chunks:
        context.append(chunk.text)
        entities = get_entities_from_chunk(chunk.id)
        for entity in entities:
            relationships = get_entity_relationships(entity.id, hops=1)
            context.append(format_relationships(relationships))

    return llm.generate(question, context=context)
```

**Cypher:**
```cypher
CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
YIELD node AS chunk, score
MATCH (chunk)<-[:FROM_CHUNK]-(entity)
MATCH (entity)-[r]-(related)
RETURN chunk.text, entity.name, type(r), related.name
ORDER BY score DESC LIMIT 5
```

#### 2. Entity-Centric Exploration

```cypher
MATCH (e:Entity {name: $name})
MATCH (e)-[:FROM_CHUNK]->(chunk)-[:HAS_CHUNK]-(doc:Document)
MATCH (e)-[r]-(related:Entity)
RETURN e, collect(DISTINCT doc) as documents,
       collect({type: type(r), entity: related}) as relationships
```

**Use case:** "Show me all documents mentioning Tesla and its relationships"

#### 3. Cross-Document Reasoning

```cypher
MATCH (a:Entity {name: $entity_a})
MATCH (b:Entity {name: $entity_b})
MATCH path = shortestPath((a)-[*..4]-(b))
MATCH (a)-[:FROM_CHUNK]->(chunk_a)-[:HAS_CHUNK]-(doc_a)
MATCH (b)-[:FROM_CHUNK]->(chunk_b)-[:HAS_CHUNK]-(doc_b)
RETURN path, doc_a.file_path, doc_b.file_path,
       [node in nodes(path) | node.name] as connection_chain
```

**Use case:** "How is Elon Musk connected to OpenAI across my documents?"

#### 4. Document Discovery

```cypher
// Find documents with overlapping entities
MATCH (source:Document {id: $doc_id})-[:HAS_CHUNK]->(chunk)
MATCH (chunk)<-[:FROM_CHUNK]-(entity)
MATCH (entity)-[:FROM_CHUNK]->(other_chunk)-[:HAS_CHUNK]-(other_doc)
WHERE other_doc.id <> $doc_id
WITH other_doc, count(DISTINCT entity) as overlap
ORDER BY overlap DESC LIMIT $top_k
RETURN other_doc, overlap
```

**Use case:** "Find documents similar to this contract"

**Unified Notebook Interface:**
```python
pipeline = DocumentGraphPipeline(driver, llm, embedder)

pipeline.ask("What does this say about Apple?")  # RAG
pipeline.explore("Apple Inc.")  # Entity exploration
pipeline.connect("Elon Musk", "Tesla")  # Cross-document
pipeline.similar_docs(doc_id)  # Discovery
```

## Error Handling & Robustness

### 1. OCR Failures

**Problem:** Low-quality scans produce gibberish

**Solution:**
- Check OCR confidence (threshold: 0.6)
- Validate text readability
- Mark low-confidence documents for manual review
- Store confidence score in Document node

### 2. LLM Extraction Failures

**Problem:** Invalid JSON or missed entities

**Solution:**
- Retry with backoff (max 3 attempts)
- Graceful degradation: create minimal entity structure
- Log failed chunks for debugging

### 3. Entity Reconciliation Conflicts

**Problem:** Uncertain if two entities are the same

**Solution:**
- Multiple matches → Use LLM disambiguation
- No clear match → Create new entity
- Log ambiguous cases for review

### 4. Neo4j Connection Issues

**Problem:** Database connection drops during processing

**Solution:**
- Checkpoint processed documents
- Allow resume from checkpoint
- Process in batches with incremental commits

### 5. Large Document Handling

**Problem:** Document too large for memory

**Solution:**
- Process in batches (max 50 chunks per batch)
- Incremental write to Neo4j
- Memory monitoring and cleanup

## Testing Strategy

### Test Pyramid

**1. Unit Tests (Fast, Many)**
- Chunk overlap verification
- Type-aware entity matching
- OCR confidence thresholds
- JSON schema validation

**2. Integration Tests (Medium Speed)**
- End-to-end extraction pipeline
- Entity reconciliation across chunks
- Neo4j graph creation
- Relationship normalization

**3. End-to-End Tests (Slow, Critical)**
- Native PDF → Query
- Scanned PDF with OCR → Query
- Multi-document cross-references
- All query modes (RAG, exploration, reasoning, discovery)

**4. Quality Tests (Manual/Periodic)**
- Entity extraction recall/precision on gold standard
- Relationship quality evaluation
- Query response accuracy

**Continuous Monitoring:**
- OCR confidence trends
- Entity extraction counts
- Reconciliation merge rates
- Query performance metrics

## Implementation Approach

**Phase 1: Core Pipeline**
1. Document ingestion (PDF + OCR)
2. Chunking with overlap
3. LLM extraction with dynamic schema
4. Basic entity reconciliation (exact + fuzzy, type-aware)
5. Neo4j graph construction

**Phase 2: Query Layer**
6. Vector search setup
7. RAG implementation
8. Entity exploration queries
9. Cross-document reasoning

**Phase 3: Polish**
10. Relationship normalization
11. Error handling and checkpointing
12. Testing and validation
13. Notebook interface refinement

## Success Criteria

- Process mixed document types (contracts, reports, articles) successfully
- Extract entities with >80% recall on test set
- Handle scanned documents with OCR confidence >70%
- Support all four query modes (RAG, exploration, reasoning, discovery)
- Graceful error handling with clear user feedback
- Modular design allowing component swaps

## Future Enhancements

- Multi-language support (extend OCR and NER)
- Real-time processing (stream documents)
- Graph visualization interface
- Entity linking to external knowledge bases (Wikipedia, DBpedia)
- Active learning for schema refinement
- Graph database abstraction layer (support NetworkX, other DBs)

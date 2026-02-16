# Document to GraphRAG

A pipeline designed to transform unstructured documents into a queryable Knowledge Graph for Retrieval-Augmented Generation (RAG) applications.

## Overview

This project bridges the gap between static documents and dynamic graph databases. By leveraging advanced OCR, PDF extraction, and Large Language Models (LLMs), it automatically extracts entities and relationships from text, enabling more context-aware and structured data retrieval.

## Features

- **Multi-Format Document Ingestion**: Seamlessly process PDFs and images using built-in extractors and OCR support (Tesseract).
- **Intelligent Entity Extraction**: Leverages OpenAI's LLMs to identify key entities and complex relationships within document chunks.
- **Dynamic Schema Inference**: Automatically infers and optimizes the graph schema based on document content.
- **Graph Database Integration**: Native support for Neo4j to store and query generated knowledge graphs.
- **Advanced RAG Capabilities**: Optimized for GraphRAG workflows, providing a structured foundation for more accurate LLM responses.
- **Customizable Pipeline**: Modular architecture allowing for easy integration of different extraction strategies and model configurations.

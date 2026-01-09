# Advanced RAG: Agentic Query Re-Writer & RRF

This project implements an Agentic RAG (Retrieval-Augmented Generation) pipeline designed to overcome the limitations of standard vector search.      
By using an LLM-based Query Re-Writer, the system transforms a single user prompt into multiple search perspectives, which are then fused using       
the Reciprocal Rank Fusion (RRF) algorithm.

## Overview

Standard RAG often fails when a user's phrasing doesn't match the document text. This agent solves that by:

* Transforming the query into three versions (Natural Language, Keyword, and Conceptual)
* Retrieving documents for all three paths locally
* Ranking the results using RRF math to find the "consensus" documents
* Synthesizing a final answer grounded in the top-ranked context

## System Configuration & Compatibility

This project is optimized for high-efficiency performance on entry-level AI hardware:

* CPU: AMD Ryzen 5 4600H (6 Cores)
* RAM: 8GB DDR4
* LLM (Inference): llama3.2:3b via Ollama
* Embedding Model: all-minilm (Lightweight/High-speed)

## Key Components

### 1. Query Re-Writer Agent

The agent takes a raw input and generates three optimized queries. This ensures that if the vector database misses a "Natural Language" match,        
it might still catch a "Keyword" or "Conceptual" match.

### 2. Reciprocal Rank Fusion (RRF)

To merge results from three different searches, we use the RRF algorithm.

#### Formula:

$$Score(d \in D) = \sum_{q \in queries} \frac{1}{k + rank(q, d)}$$

We use $k=60$ as a smoothing constant to ensure stable rankings and prevent any single query from dominating the results.

## Evaluation Metric: Mean Reciprocal Rank (MRR)

We evaluate the system using Mean Reciprocal Rank, which measures how high up the "gold standard" document appears in our results.

## Test Results

| Question | Queries Used | Found at Rank | Reciprocal Rank (RR) | RRF Math Formula | Final MRR |
| --- | --- | --- | --- | --- | --- |
| Ryzen 5 Core Count311.003B Model RAM (4-bit) | 311.00 | 311.00 | 311.00 | 311.00 | 1.00 |

## Project Structure

```markdown
├── main.py            # Main execution & Evaluation logic
├── rewriter.py        # LLM-based Query Transformation
├── retriever.py       # ChromaDB setup & RRF scoring function
├── .gitignore         # Excludes .venv and Chroma database
└── README.md          # Project documentation
```

## Getting Started

Ensure Ollama is running with llama3.2:3b and all-minilm.

Run the pipeline using:
```bash
PowerShell python main.py python evaulation.py
```
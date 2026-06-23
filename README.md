# Generalized Glossary Extraction Pipeline

Multi-domain, PDF-aware glossary extraction using Groq (Llama) as the LLM backend.

---

## File Structure

```
glossary_pipeline/
├── config.py          # All constants, domain registry, category map
├── models.py          # Pydantic schemas for strict response validation
├── pdf_reader.py      # PDF → text extraction with quality checks
├── chunker.py         # Preprocessing + paragraph chunking with overlap
├── llm_client.py      # LLM call (single call: domain + glossary), retry logic
├── postprocessor.py   # Validate, normalize, deduplicate across chunks
├── pipeline.py        # Orchestrator — ties all modules together
└── demo.py            # Sample input/output + edge-case assertions
```

---

## Quick Start

```bash
pip install groq pdfplumber pydantic
python demo.py
```

---

## Architecture

```
Input (PDF paths or raw text + dataset_id)
          │
          ▼
  pdf_reader.py          Quality checks: encrypted? scanned? empty?
          │
          ▼
  chunker.py             preprocess() → chunk_text() with overlap + token cap
          │
          ▼ (per chunk)
  llm_client.py          Single LLM call → { domain, confidence, glossary[] }
          │                Retry w/ backoff on 429/503
          │
          ▼ (after all chunks)
  postprocessor.py       validate → normalize (domain-aware) → deduplicate
          │                Near-duplicate logging (Jaccard ≥ 0.7)
          ▼
  pipeline.py            Aggregates per-doc results into DatasetResult
          │                Dataset-level domain cache
          │                Per-doc failure isolation
          ▼
  Output JSON
```

---

## Input / Output

### Input
```json
{
  "dataset_id": "legal_contracts_q4",
  "documents": [
    { "doc_id": "contract_01", "pdf_path": "/data/contract_01.pdf" },
    { "doc_id": "contract_02", "content": "raw text fallback..." }
  ]
}
```

### Output
```json
{
  "dataset_id": "legal_contracts_q4",
  "total_docs": 2,
  "successful_docs": 2,
  "flagged_docs": 0,
  "results": [
    {
      "doc_id": "contract_01",
      "detected_domain": "law",
      "confidence": "high",
      "flagged": false,
      "error": null,
      "glossary": [
        {
          "term": "indemnification_clause",
          "definition": "A contractual provision that holds one party harmless from liability arising from the other party's actions.",
          "category": "liability"
        }
      ]
    }
  ]
}
```

---

## Supported Domains

Defined in `config.py` — add new domains by extending `DOMAIN_CATEGORIES`:

| Domain      | Example Categories                                  |
|-------------|-----------------------------------------------------|
| airline     | boarding, flight_ops, airport, safety, crew         |
| medicine    | diagnosis, treatment, anatomy, pharmacology         |
| law         | contract, procedure, evidence, liability            |
| finance     | instruments, risk, regulation, valuation            |
| technology  | architecture, protocol, security, data              |
| engineering | mechanics, materials, thermodynamics, control       |
| general     | concept, process, entity, attribute (fallback)      |

---

## Key Design Decisions

### Single LLM call per chunk
Domain detection and glossary extraction happen in one call. The model returns `{"domain": "...", "confidence": "...", "glossary": [...]}`. Eliminates the 2-call pattern from the previous design — saves ~50% LLM cost.

### Domain locked after first chunk
Domain is detected from chunk 1 of each document and cached for all subsequent chunks. Avoids repeated detection. If confidence is below `DOMAIN_CONFIDENCE_THRESHOLD` (0.6), falls back to `"general"` for all chunks.

### Dataset-level domain cache
After the first high-confidence document in a dataset resolves its domain, subsequent documents in the same dataset skip detection entirely. Ideal for homogeneous datasets (e.g., all legal contracts).

### Paragraph chunking with overlap
Chunks respect paragraph boundaries. Adjacent chunks share `CHUNK_OVERLAP` paragraphs so terms defined across a paragraph boundary are never lost.

### Near-duplicate logging (not auto-merging)
Terms with Jaccard token overlap ≥ 0.7 are logged as warnings. They are not auto-merged — `blood_pressure` and `arterial_blood_pressure` look similar but may have distinct definitions across chunks.

### Per-document failure isolation
One failed PDF or LLM error never blocks the rest of the batch. Each document returns its own `error` field.

---

## Extending the Pipeline

**Add a new domain:**
```python
# config.py
DOMAIN_CATEGORIES["logistics"] = [
    "shipment", "route", "inventory", "carrier", "customs", "warehouse"
]
```

**Swap the LLM:**
Replace `call_llm_for_chunk()` in `llm_client.py` with any function that returns a `ChunkLLMResponse`.

**Add OCR for scanned PDFs:**
In `pdf_reader.py`, when `scanned_pages / total_pages > 0.8`, call `pytesseract.image_to_string()` per page instead of returning an error.

**Parallel processing (production):**
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(lambda doc: process_document(...), documents))
```

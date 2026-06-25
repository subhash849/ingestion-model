"""
document_parser.py — Parse documents into structured sections, subsections, and summaries.

Usage:
  python document_parser.py                        # process all PDFs in documents/
  python document_parser.py --file path/to/file.pdf # process a single file

Output: JSON files saved to outputs/ with hierarchical section structure.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from chunker import preprocess, chunk_text

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("documents")
OUTPUTS_DIR = Path("outputs")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "minimax-m3:cloud")
MAX_TOKENS = 4096

def extract_text_from_pdf(path: str | Path) -> tuple[str, str | None]:
    try:
        import pdfplumber
    except ImportError:
        return "", "pdfplumber not installed"

    path = Path(path)
    if not path.exists():
        return "", f"File not found: {path}"
    if path.suffix.lower() != ".pdf":
        return "", f"Not a PDF file: {path}"

    try:
        with pdfplumber.open(path) as pdf:
            pages_text: list[str] = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"[Page {i+1}]\n{text}")
            full_text = "\n\n".join(pages_text).strip()
            if not full_text:
                return "", f"No text could be extracted from: {path.name}"
            logger.info("Extracted %d chars from %d pages of '%s'", len(full_text), len(pages_text), path.name)
            return full_text, None
    except Exception as exc:
        return "", f"Failed to read PDF '{path.name}': {exc}"


def _build_system_prompt() -> str:
    return """You are an expert document analyst. Your task is to parse the given document text into a structured hierarchy of sections and subsections with summaries.

For each major section you identify, extract:
1. The section title
2. A brief summary of the section (1-3 sentences)
3. Any subsections within it, each with:
   - The subsection title
   - A summary or the key important points of that subsection (bullet points preferred)

Rules:
- Identify sections based on headings, numbered sections, or thematic breaks in the text.
- Do NOT invent content — only extract what is present in the text.
- Use the [Page X] markers to reference page numbers where relevant.
- Keep summaries concise and factual.

Respond ONLY with a valid JSON object — no markdown fences, no prose:
{
  "document_summary": "Brief 1-2 sentence summary of the entire document",
  "sections": [
    {
      "section_title": "string",
      "section_summary": "string",
      "page_number": "string or null",
      "subsections": [
        {
          "subsection_title": "string",
          "summary": "string (paragraph or bullet points of key information)",
          "key_points": ["point 1", "point 2", "..."],
          "page_number": "string or null"
        }
      ]
    }
  ]
}"""


def call_llm(text: str, retries: int = 3) -> dict | None:
    system_prompt = _build_system_prompt()
    user_message = f"Parse the following document text into sections and subsections:\n\n{text}"

    for attempt in range(1, retries + 1):
        try:
            if LLM_PROVIDER == "ollama":
                import ollama
                client = ollama.Client(host=OLLAMA_HOST)
                response = client.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    options={"num_predict": MAX_TOKENS},
                )
                raw = response["message"]["content"].strip()
            else:
                from groq import Groq
                client = Groq(api_key=GROQ_API_KEY)
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                raw = response.choices[0].message.content.strip()

            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)
            return data

        except Exception as exc:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            return None

    return None


def _merge_results(results: list[dict]) -> dict:
    merged_sections: list[dict] = []
    seen_sections: dict[str, dict] = {}

    for result in results:
        sections = result.get("sections", [])
        for section in sections:
            title = section.get("section_title", "").strip().lower()
            if title in seen_sections:
                existing = seen_sections[title]
                existing_subs = {s.get("subsection_title", "").strip().lower(): s for s in existing.get("subsections", [])}
                for sub in section.get("subsections", []):
                    sub_title = sub.get("subsection_title", "").strip().lower()
                    if sub_title and sub_title not in existing_subs:
                        existing.setdefault("subsections", []).append(sub)
                    elif sub_title:
                        logger.debug("Duplicate subsection '%s' in section '%s' merged", sub.get("subsection_title"), section.get("section_title"))
            else:
                merged_sections.append(section)
                seen_sections[title] = section

    summary = ""
    for r in results:
        if r.get("document_summary"):
            summary = r["document_summary"]
            break

    return {
        "document_summary": summary,
        "sections": merged_sections,
    }


def process_document(doc_id: str, pdf_path: Path) -> dict | None:
    logger.info("Processing: %s", pdf_path.name)

    text, error = extract_text_from_pdf(str(pdf_path))
    if error:
        logger.error("Failed to extract text from '%s': %s", pdf_path.name, error)
        return None

    text = preprocess(text)
    chunks = chunk_text(text)
    logger.info("Split into %d chunks", len(chunks))

    all_results: list[dict] = []
    for i, chunk in enumerate(chunks):
        logger.info("Processing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        result = call_llm(chunk)
        if result:
            all_results.append(result)
        else:
            logger.warning("Chunk %d returned no result — skipping", i + 1)

    if not all_results:
        logger.error("No results obtained from any chunk for '%s'", pdf_path.name)
        return None

    merged = _merge_results(all_results)
    return {
        "document_id": doc_id,
        "source_file": pdf_path.name,
        **merged,
    }


def run(target_file: str | None = None):
    print("=" * 65)
    print("  Document Parser — Section/Subsection Extraction")
    print("=" * 65)

    DOCUMENTS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set in .env")
        return
    if LLM_PROVIDER not in ("groq", "ollama"):
        print(f"Error: Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' or 'ollama'.")
        return

    print(f"LLM Provider: {LLM_PROVIDER}")

    if target_file:
        pdf_path = Path(target_file)
        if not pdf_path.exists():
            print(f"Error: File '{pdf_path}' not found.")
            return
        pdf_files = [pdf_path]
    else:
        pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {DOCUMENTS_DIR.absolute()}")
            print("Place PDFs there or use --file to specify one.")
            return

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        result = process_document(doc_id, pdf_path)
        if result is None:
            print(f"  ✗ Failed to process: {pdf_path.name}")
            continue

        out_file = OUTPUTS_DIR / f"{doc_id}_parsed.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        section_count = len(result.get("sections", []))
        print(f"  ✓ {pdf_path.name} — {section_count} sections extracted")
        print(f"    Output: {out_file.absolute()}")

    print(f"\n{'='*65}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDF documents into structured sections")
    parser.add_argument("--file", "-f", type=str, help="Path to a specific PDF file to process")
    args = parser.parse_args()
    run(target_file=args.file)

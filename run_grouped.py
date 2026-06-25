import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

DOCUMENTS_DIR = Path("documents")
OUTPUTS_DIR = Path("outputs")

def run_grouped(target_file: str | None = None):
    print("=" * 65)
    print("  Generalized Glossary Extraction Pipeline (Grouped)")
    print("=" * 65)

    DOCUMENTS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if target_file:
        pdf_path = Path(target_file)
        if not pdf_path.exists():
            print(f"Error: File '{pdf_path}' not found.")
            return
        pdf_files = [pdf_path]
        dataset_id = f"single_run_grouped_{pdf_path.stem}"
    else:
        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
        dataset_id = "batch_run_grouped_001"
        if not pdf_files:
            print(f"No PDF files found in {DOCUMENTS_DIR.absolute()}")
            print("Please place some PDF files there and run again, or use --file to specify a file.")
            return

    documents_payload = []
    for pdf_path in pdf_files:
        documents_payload.append({
            "doc_id": pdf_path.stem,
            "pdf_path": str(pdf_path)
        })

    payload = {
        "dataset_id": dataset_id,
        "documents": documents_payload
    }

    output = run_pipeline(payload)

    # Transform the output into grouped format
    grouped_results = []
    for result in output["results"]:
        glossary_list = result["glossary"]
        
        grouped_glossary = defaultdict(list)
        for entry in glossary_list:
            # Handle both dictionary and object
            if isinstance(entry, dict):
                cat = entry.get("category", "general")
                grouped_glossary[cat].append({
                    "term": entry.get("term"),
                    "definition": entry.get("definition"),
                    "page_number": entry.get("page_number"),
                    "chunk_number": entry.get("chunk_number")
                })
            else:
                cat = getattr(entry, "category", "general")
                grouped_glossary[cat].append({
                    "term": getattr(entry, "term", None),
                    "definition": getattr(entry, "definition", None),
                    "page_number": getattr(entry, "page_number", None),
                    "chunk_number": getattr(entry, "chunk_number", None)
                })
        
        grouped_result = {
            "doc_id": result["doc_id"],
            "detected_domain": result["detected_domain"],
            "confidence": result["confidence"],
            "flagged": result["flagged"],
            "error": result.get("error"),
            "glossary_by_category": dict(grouped_glossary)
        }
        grouped_results.append(grouped_result)

    grouped_output = {
        "dataset_id": output["dataset_id"],
        "total_docs": output["total_docs"],
        "successful_docs": output["successful_docs"],
        "flagged_docs": output["flagged_docs"],
        "results": grouped_results
    }

    # Write full JSON to outputs folder
    out_file = OUTPUTS_DIR / f"{dataset_id}_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(grouped_output, f, indent=2, default=str)
    
    # ── Pretty summary ─────────────────────────────────────────────────────
    print(f"\nDataset  : {grouped_output['dataset_id']}")
    print(f"Total    : {grouped_output['total_docs']} docs | "
          f"Success: {grouped_output['successful_docs']} | "
          f"Flagged: {grouped_output['flagged_docs']}")

    for result in grouped_output["results"]:
        doc_id   = result["doc_id"]
        domain   = result["detected_domain"]
        conf     = result["confidence"]
        flagged  = result["flagged"]
        error    = result.get("error")
        glossary_by_cat = result["glossary_by_category"]

        flag_str = " ⚑ FLAGGED" if flagged else ""
        print(f"\n{'─'*60}")
        print(f"📄 {doc_id}{flag_str}")
        print(f"   Domain: {domain}  |  Confidence: {conf}")

        if error:
            print(f"   ✗ Error: {error}")
        elif glossary_by_cat:
            term_count = sum(len(terms) for terms in glossary_by_cat.values())
            print(f"   {term_count} term(s) extracted across {len(glossary_by_cat)} categories:")
            for cat, terms in glossary_by_cat.items():
                print(f"\n   📁 Category: [{cat.upper()}]")
                for term_data in terms:
                    term = term_data["term"]
                    defn = term_data["definition"]
                    page = term_data["page_number"]
                    chunk = term_data["chunk_number"]
                    
                    meta_str = []
                    if page: meta_str.append(f"Page: {page}")
                    if chunk: meta_str.append(f"Chunk: {chunk}")
                    meta_tag = f" [{', '.join(meta_str)}]" if meta_str else ""
                    
                    print(f"     • {meta_tag} {term}")
                    print(f"         {defn}")
        else:
            print("   (no terms extracted)")

    print(f"\n{'='*65}")
    print(f"Output saved to {out_file.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Glossary Extraction Pipeline (Grouped by Category)")
    parser.add_argument("--file", "-f", type=str, help="Path to a specific PDF file to process")
    args = parser.parse_args()

    run_grouped(target_file=args.file)

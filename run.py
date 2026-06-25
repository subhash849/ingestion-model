import argparse
import json
import logging
from pathlib import Path
from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

DOCUMENTS_DIR = Path("documents")
OUTPUTS_DIR = Path("outputs")

def run_demo(target_file: str | None = None):
    print("=" * 65)
    print("  Generalized Glossary Extraction Pipeline")
    print("=" * 65)

    # Ensure directories exist
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if target_file:
        pdf_path = Path(target_file)
        if not pdf_path.exists():
            print(f"Error: File '{pdf_path}' not found.")
            return
        pdf_files = [pdf_path]
        dataset_id = f"single_run_{pdf_path.stem}"
    else:
        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
        dataset_id = "batch_run_001"
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

    # Write full JSON to outputs folder
    out_file = OUTPUTS_DIR / f"{dataset_id}_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    
    # ── Pretty summary ─────────────────────────────────────────────────────
    print(f"\nDataset  : {output['dataset_id']}")
    print(f"Total    : {output['total_docs']} docs | "
          f"Success: {output['successful_docs']} | "
          f"Flagged: {output['flagged_docs']}")

    for result in output["results"]:
        doc_id   = result["doc_id"]
        domain   = result["detected_domain"]
        conf     = result["confidence"]
        flagged  = result["flagged"]
        error    = result.get("error")
        glossary = result["glossary"]

        flag_str = " ⚑ FLAGGED" if flagged else ""
        print(f"\n{'─'*60}")
        print(f"📄 {doc_id}{flag_str}")
        print(f"   Domain: {domain}  |  Confidence: {conf}")

        if error:
            print(f"   ✗ Error: {error}")
        elif glossary:
            print(f"   {len(glossary)} term(s) extracted:")
            for entry in glossary:
                # Handle both dict and object
                term = entry["term"] if isinstance(entry, dict) else entry.term
                defn = entry["definition"] if isinstance(entry, dict) else entry.definition
                cat  = entry["category"] if isinstance(entry, dict) else entry.category
                page = entry.get("page_number") if isinstance(entry, dict) else getattr(entry, "page_number", None)
                chunk = entry.get("chunk_number") if isinstance(entry, dict) else getattr(entry, "chunk_number", None)
                
                meta_str = []
                if page: meta_str.append(f"Page: {page}")
                if chunk: meta_str.append(f"Chunk: {chunk}")
                meta_tag = f" [{', '.join(meta_str)}]" if meta_str else ""
                
                print(f"     • [{cat}]{meta_tag} {term}")
                print(f"         {defn}")
        else:
            print("   (no terms extracted)")

    print(f"\n{'='*65}")
    print(f"Output saved to {out_file.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Glossary Extraction Pipeline")
    parser.add_argument("--file", "-f", type=str, help="Path to a specific PDF file to process (e.g. documents/my_file.pdf)")
    args = parser.parse_args()

    run_demo(target_file=args.file)

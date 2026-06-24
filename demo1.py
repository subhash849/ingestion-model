"""
demo1.py — Test the glossary pipeline against Section_6.pdf

Run:
    python3 demo1.py
"""

import json
import logging
from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

SAMPLE_PAYLOAD = {
    "dataset_id": "airline_ops_test",
    "documents": [
        {
            "doc_id": "Section_6",
            "pdf_path": "Section_6.pdf"
        }
    ]
}


def run_demo():
    print("=" * 65)
    print("  Glossary Extraction  ")
    print("=" * 65)

    output = run_pipeline(SAMPLE_PAYLOAD)

    # ── Summary header ──────────────────────────────────────────────────────
    print(f"\nDataset     : {output['dataset_id']}")
    print(f"Total docs  : {output['total_docs']}")
    print(f"Successful  : {output['successful_docs']}")
    print(f"Flagged     : {output['flagged_docs']}")

    for result in output["results"]:
        doc_id   = result["doc_id"]
        domain   = result["detected_domain"]
        conf     = result["confidence"]
        flagged  = result["flagged"]
        error    = result.get("error")
        glossary = result["glossary"]

        flag_str = "  ⚑ FLAGGED" if flagged else ""
        print(f"\n{'─' * 65}")
        print(f"📄 {doc_id}{flag_str}")
        print(f"   Domain     : {domain}")
        print(f"   Confidence : {conf}")

        if error:
            print(f"\n   ✗ Error: {error}")

        elif not glossary:
            print("\n   (no terms extracted)")

        else:
            print(f"\n   {len(glossary)} term(s) extracted:\n")
            # Group by category for readable output
            by_category = {}
            for entry in glossary:
                term = entry["term"] if isinstance(entry, dict) else entry.term
                defn = entry["definition"] if isinstance(entry, dict) else entry.definition
                cat  = entry["category"] if isinstance(entry, dict) else entry.category
                by_category.setdefault(cat, []).append((term, defn))

            for cat in sorted(by_category):
                print(f"   [{cat.upper()}]")
                for term, defn in by_category[cat]:
                    print(f"     • {term}")
                    print(f"       {defn}")
                print()

    # ── Full JSON ────────────────────────────────────────────────────────────
    print("=" * 65)
    print("Full JSON output:\n")
    print(json.dumps(output, indent=2, default=str))

    # ── Assertions ───────────────────────────────────────────────────────────
    for result in output["results"]:
        if result.get("error"):
            continue
        for entry in result["glossary"]:
            term = entry["term"] if isinstance(entry, dict) else entry.term
            defn = entry["definition"] if isinstance(entry, dict) else entry.definition
            cat  = entry["category"] if isinstance(entry, dict) else entry.category
            assert " " not in term,       f"FAIL: term '{term}' contains spaces"
            assert term == term.lower(),  f"FAIL: term '{term}' is not lowercase"
            assert defn,                  f"FAIL: term '{term}' has empty definition"
            assert cat,                   f"FAIL: term '{term}' has empty category"

    print("\n All assertions passed.")


if __name__ == "__main__":
    run_demo()

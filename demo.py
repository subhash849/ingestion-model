"""
demo.py — Sample usage for the upgraded glossary extraction pipeline.

Run:
    pip install groq pdfplumber pydantic
    python demo.py
"""

import json
import logging
from pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Sample payload — text-based (no PDF needed to run this demo)
# To use PDFs, replace "content" with "pdf_path": "/path/to/file.pdf"
# ---------------------------------------------------------------------------

SAMPLE_PAYLOAD = {
    "dataset_id": "demo_batch_001",
    "documents": [
        # ── Doc 1: Airline domain ──────────────────────────────────────────
        {
            "doc_id": "airline_ops_manual",
            "content": (
                "Passengers flying on the morning service should proceed to the gate "
                "at least 30 minutes before departure. The boarding pass can be obtained "
                "at the check-in counter or via the airline's mobile app. "
                "Overhead bins fill quickly, so passengers with carry-on luggage are "
                "encouraged to board during early boarding. Seat upgrades to business "
                "class are available at the gate subject to availability. The cabin crew "
                "will demonstrate the emergency exits and safety procedures before "
                "the aircraft reaches cruising altitude.\n\n"
                "After landing, passengers should remain seated until the fasten seatbelt "
                "sign is switched off. Checked baggage can be collected at the baggage "
                "carousel in the arrivals hall. Passengers connecting to an onward flight "
                "must proceed through the transit area and present their boarding pass "
                "at the transfer desk."
            ),
        },

        # ── Doc 2: Medical domain ──────────────────────────────────────────
        {
            "doc_id": "medical_guidelines",
            "content": (
                "Informed consent must be obtained before any surgical procedure. "
                "The attending physician is responsible for explaining the diagnosis, "
                "treatment options, and associated risks to the patient.\n\n"
                "Post-operative care includes monitoring vital signs such as blood pressure, "
                "heart rate, and oxygen saturation. Analgesics are administered to manage "
                "pain during the recovery period. Patients with comorbidities such as "
                "hypertension or diabetes require special monitoring protocols.\n\n"
                "Diagnostic imaging including MRI and CT scans are used to assess "
                "internal structures. A biopsy may be performed when malignancy is suspected."
            ),
        },

        # ── Doc 3: Legal domain ────────────────────────────────────────────
        {
            "doc_id": "contract_terms",
            "content": (
                "This agreement constitutes the entire contract between the parties and "
                "supersedes all prior negotiations and representations. The indemnification "
                "clause holds each party harmless from liability arising out of the other "
                "party's negligence.\n\n"
                "Any dispute arising under this agreement shall be subject to binding "
                "arbitration under the rules of the relevant jurisdiction. The prevailing "
                "party in any arbitration proceeding shall be entitled to recover reasonable "
                "attorney fees and court costs.\n\n"
                "Force majeure provisions excuse performance when circumstances beyond "
                "the reasonable control of a party prevent fulfillment of contractual obligations."
            ),
        },

        # ── Doc 4: Edge case — empty content ──────────────────────────────
        {
            "doc_id": "empty_doc",
            "content": "",
        },

        # ── Doc 5: Edge case — noise/raw values only ──────────────────────
        {
            "doc_id": "noise_only",
            "content": "AI202 JFK BOM 14:30 1/5/2025 !!!",
        },
    ]
}


def run_demo():
    print("=" * 65)
    print("  Generalized Glossary Extraction Pipeline — Demo Run")
    print("=" * 65)

    output = run_pipeline(SAMPLE_PAYLOAD)

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
                print(f"     • [{cat}] {term}")
                print(f"         {defn}")
        else:
            print("   (no terms extracted)")

    print(f"\n{'='*65}")
    print("Full JSON output:")
    print(json.dumps(output, indent=2, default=str))

    # ── Assertions ────────────────────────────────────────────────────────
    by_id = {r["doc_id"]: r for r in output["results"]}

    assert by_id["empty_doc"]["glossary"] == [],      "FAIL: empty doc should have no terms"
    assert by_id["noise_only"]["glossary"] == [],     "FAIL: noise doc should have no terms"
    assert by_id["empty_doc"]["flagged"] is True,     "FAIL: empty doc should be flagged"

    for r in output["results"]:
        for entry in r["glossary"]:
            term = entry["term"] if isinstance(entry, dict) else entry.term
            defn = entry["definition"] if isinstance(entry, dict) else entry.definition
            cat  = entry["category"] if isinstance(entry, dict) else entry.category
            assert " " not in term,  f"FAIL: term '{term}' contains spaces"
            assert term == term.lower(), f"FAIL: term '{term}' is not lowercase"
            assert defn,  f"FAIL: term '{term}' has empty definition"
            assert cat,   f"FAIL: term '{term}' has empty category"

    print("\n All assertions passed.")


if __name__ == "__main__":
    run_demo()

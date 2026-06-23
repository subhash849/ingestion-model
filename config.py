"""
config.py — Central configuration for the glossary extraction pipeline.
All tuneable constants live here. Nothing else imports from sibling modules.
"""

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
import os


LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # "groq" or "ollama"

# Groq Config
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.1-8b-instant"

# Ollama Config
OLLAMA_HOST   = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL", "phi4-mini:latest")

MAX_TOKENS    = 2048

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_MAX_TOKENS  = 400   # hard ceiling per chunk (safety guard)
CHUNK_OVERLAP     = 1     # number of paragraphs to repeat at chunk boundaries

# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

DOMAIN_CONFIDENCE_THRESHOLD = 0.6   # below this → fall back to "general"

# Supported domains and their category lists.
# Add a new domain here — the rest of the pipeline picks it up automatically.
DOMAIN_CATEGORIES: dict[str, list[str]] = {
    "airline":    ["boarding", "flight_ops", "airport", "safety",
                   "ticketing", "crew", "baggage", "amenities", "navigation"],
    "medicine":   ["diagnosis", "treatment", "anatomy", "pharmacology",
                   "procedure", "imaging", "pathology", "clinical"],
    "law":        ["contract", "procedure", "evidence", "parties",
                   "jurisdiction", "remedy", "statute", "liability"],
    "finance":    ["instruments", "risk", "regulation", "valuation",
                   "accounting", "market", "credit", "compliance"],
    "technology": ["architecture", "protocol", "security", "data",
                   "networking", "interface", "algorithm", "infrastructure"],
    "engineering":["mechanics", "materials", "thermodynamics", "control",
                   "electrical", "structural", "manufacturing", "measurement"],
    "general":    ["concept", "process", "entity", "attribute",
                   "relationship", "metric", "role", "system"],
}

SUPPORTED_DOMAINS = list(DOMAIN_CATEGORIES.keys())   # excludes "general"
FALLBACK_DOMAIN   = "general"

# ---------------------------------------------------------------------------
# Raw-value patterns (never extract these as terms)
# ---------------------------------------------------------------------------

RAW_VALUE_PATTERNS: list[str] = [
    r"\b[A-Z]{2}\d{3,4}\b",           # Flight / serial numbers: AI202
    r"\b\d{1,2}[A-F]\b",              # Seat codes: 12A
    r"\b\d{2}:\d{2}\b",               # Times: 14:30
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates: 01/05/2024
]

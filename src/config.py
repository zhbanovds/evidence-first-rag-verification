from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CONFLICT_DIR = DATA_DIR / "raw_conflict"
EVAL_DIR = DATA_DIR / "eval"
PROCESSED_DIR = DATA_DIR / "processed"

QUESTIONS_MAIN_PATH = EVAL_DIR / "questions_main.csv"
QUESTIONS_CONFLICT_PATH = EVAL_DIR / "questions_conflict.csv"
EXPECTED_CLAIMS_PATH = EVAL_DIR / "expected_claims.json"
CONFLICT_METADATA_PATH = EVAL_DIR / "conflict_metadata.csv"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_CLEAN_DIR = CHROMA_DIR / "clean"
CHROMA_CONFLICT_DIR = CHROMA_DIR / "conflict"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_MAIN_DIR = RESULTS_DIR / "main"
RESULTS_CONFLICT_DIR = RESULTS_DIR / "conflict"

LLM_MODEL = "qwen3.5:9b"
FALLBACK_LLM_MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / ".cache" / "sentence_transformers"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

TOP_K = 5
VECTOR_TOP_N = 10
BM25_TOP_N = 10
RRF_K = 60

TEMPERATURE = 0.1
OLLAMA_TIMEOUT_SEC = 600

FACT_PLAN_MAX_CLAIMS = 7
MAX_CLAIMS_TO_VERIFY = 10

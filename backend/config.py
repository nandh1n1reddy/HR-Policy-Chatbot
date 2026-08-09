from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = BASE_DIR / "documents"
CHROMA_DB_PATH = BASE_DIR / "chromadb_data"
AUDIT_DB_PATH = BASE_DIR / "hr_policy_audit.sqlite3"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Cross-encoder used to rerank the top vector-search candidates. A cross-encoder
# scores the query and passage together, which is far more accurate than
# comparing two independently-embedded vectors, so this is what makes the
# confidence score trustworthy enough to run a strict 0.75 gate against.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hr_policy_chunks")

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "mistral")
CLOUD_LLM_MODEL = os.getenv("CLOUD_LLM_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Caps how many tokens the local Ollama model is allowed to generate. The
# writer only needs a short JSON answer, so an uncapped local model can spend
# far longer than necessary generating. Lower this further (e.g. 250) for a
# faster but more clipped local answer.
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "400"))
# The heuristic keyword classifier in Agents/intent_classifier.py is fast and
# usually good enough for this fixed set of HR categories. Calling the local
# LLM just to classify intent adds a full extra Ollama round trip before
# retrieval even starts, so it's opt-in rather than the default.
USE_LLM_INTENT_CLASSIFIER = os.getenv("USE_LLM_INTENT_CLASSIFIER", "false").lower() == "true"

# Number of chunks handed to the writer LLM. More chunks can improve
# confidence but also means more context for the writer LLM to process, which
# is the single slowest step in the pipeline -- keep this modest.
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "7"))
# Anything scoring below this is queued for human review instead of being
# auto-answered. 0.75 is intentionally strict: pair it with RERANK_MODEL
# above and the writer's confidence rubric (see Agents/writer.py) rather
# than lowering this number to reduce escalations.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
# Smaller, more topic-pure chunks make each retrieved excerpt easier for both
# the cross-encoder and the writer LLM to judge as relevant or not.
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
# Kept moderate on purpose: this is the writer LLM's input size, and a larger
# context window directly slows down generation, especially on local CPU
# inference.
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
# Caps how many query variants SearchEngine tries per question (base query +
# intent-prefixed + keyword-augmented variants). Each variant is a real
# vector search call, so this is a direct lever on retrieval latency.
MAX_QUERY_VARIANTS = int(os.getenv("MAX_QUERY_VARIANTS", "3"))

INTENT_CATEGORY_HINTS = {
    "leave": ["Leave", "Attendance", "Employee_Handbook"],
    "wfh": ["WFH", "Remote_Work"],
    "overtime_comp_off": ["Compensation", "Payroll"],
    "whistleblower": ["Grievance", "Employee_Handbook"],
    "harassment_grievance": ["POSH", "Grievance", "International"],
    "anti_bribery": ["Code_of_Conduct", "Employee_Handbook"],
    "ethics_conduct": ["Code_of_Conduct", "Employee_Handbook"],
    "data_privacy": ["Data_Privacy", "IT_Security"],
    "travel_expense": ["Travel", "Compensation"],
    "disciplinary": ["Employee_Handbook", "Grievance", "Code_of_Conduct"],
    "general_hr": [],
}

INTENT_QUERY_HINTS = {
    "leave": ["casual leave", "earned leave", "sick leave", "annual leave", "restricted holiday"],
    "wfh": ["work from home", "remote work", "telework", "work from home policy"],
    "overtime_comp_off": ["overtime", "comp off", "compensatory off", "holiday work"],
    "whistleblower": ["whistleblower", "anonymous complaint", "report misconduct"],
    "harassment_grievance": ["harassment", "grievance", "posh", "discrimination"],
    "anti_bribery": ["bribery", "corruption", "gift policy"],
    "ethics_conduct": ["ethics", "code of conduct", "professional conduct"],
    "data_privacy": ["data privacy", "confidentiality", "personal data"],
    "travel_expense": ["travel expense", "reimbursement", "expense claim"],
    "disciplinary": ["disciplinary", "warning", "termination", "misconduct"],
    "general_hr": [],
}

os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(AUDIT_DB_PATH.parent, exist_ok=True)

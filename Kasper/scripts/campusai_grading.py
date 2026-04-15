import json
import os
import random
import re
import time
import gc
import csv
import argparse

import pymupdf as fitz
from openai import OpenAI
import sys

try:
    from google.cloud import storage
except ImportError:
    storage = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    # Load both cwd .env and script-local .env when present.
    load_dotenv()
    script_env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(script_env_path, override=False)

# 1. Setup Client (OpenAI-compatible API)
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "14000"))
CHARS_PER_TOKEN_ESTIMATE = float(os.getenv("CHARS_PER_TOKEN_ESTIMATE", "3.3"))
REQUEST_TEMPERATURE = float(os.getenv("REQUEST_TEMPERATURE", "0.5"))
REQUEST_TOP_P = float(os.getenv("REQUEST_TOP_P", "0.95"))
COMPACT_INPUT_WHITESPACE = os.getenv("COMPACT_INPUT_WHITESPACE", "0") == "0"
CANARY_MAX_FILES = int(os.getenv("CANARY_MAX_FILES", "0"))
STORE_FULL_ANALYSIS = os.getenv("STORE_FULL_ANALYSIS", "0") == "1"
ANALYSIS_MAX_CHARS = int(os.getenv("ANALYSIS_MAX_CHARS", "8000"))
FORCE_GC_EACH_FILE = os.getenv("FORCE_GC_EACH_FILE", "0") == "1"
SHRINK_FITZ_STORE_PERCENT = int(os.getenv("SHRINK_FITZ_STORE_PERCENT", "100"))
PAUSE_BETWEEN_FILES_SECONDS = float(os.getenv("PAUSE_BETWEEN_FILES_SECONDS", "4"))
PRIORITIZE_SMALLEST_FROM_CSV = os.getenv("PRIORITIZE_SMALLEST_FROM_CSV", "1") == "1"
PRIORITY_FILE_LIMIT = int(os.getenv("PRIORITY_FILE_LIMIT", "0"))
PRIORITY_START_INDEX = int(os.getenv("PRIORITY_START_INDEX", "0"))
PRIORITY_END_INDEX = int(os.getenv("PRIORITY_END_INDEX", "2000"))
PDF_CHAR_COUNT_CSV = os.getenv(
    "PDF_CHAR_COUNT_CSV",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pdf_char_count_sorted.csv")),
)
GRADING_PRESET_PATH = os.getenv(
    "GRADING_PRESET_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Grading optimized for MoE.preset.json")),
)
RESEARCH_QUESTIONS_JSONL = os.getenv(
    "RESEARCH_QUESTIONS_JSONL",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extracted_research_questions.jsonl")),
)
ENABLE_STREAM_THINKING = os.getenv("ENABLE_STREAM_THINKING", "1") == "1"
USE_BATCH_API = os.getenv("USE_BATCH_API", "0") == "1"
API_BASE_URL = os.getenv("OPENAI_BASE_URL", os.getenv("CAMPUSAI_BASE_URL", "https://api.campusai.compute.dtu.dk/v1"))
API_KEY = os.getenv("CAMPUSAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Missing API key. Set OPENAI_API_KEY or CAMPUSAI_API_KEY in environment or .env file."
    )

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
    timeout=REQUEST_TIMEOUT_SECONDS,
)

# 2. Configuration
PDF_FOLDER = r"C:\Users\kkrus\OneDrive - Danmarks Tekniske Universitet\DTU\MScProject\thesis_pdfs"
PDF_INPUT_SOURCE = os.getenv("PDF_INPUT_SOURCE", "gcs").strip().lower()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "thesis_archive_bucket").strip()
GCS_PDF_PREFIX = os.getenv("GCS_PDF_PREFIX", "dtu_findit/master_thesis").strip()
OUTPUT_FILE = "data/grading_results.jsonl"
THINKING_LOG_FILE = os.getenv("THINKING_LOG_FILE", "data/grading_thinking.jsonl")
LOG_MODEL_THINKING = os.getenv("LOG_MODEL_THINKING", "1") == "1"
THINKING_MAX_CHARS = int(os.getenv("THINKING_MAX_CHARS", "40000"))
SKIP_ON_BAD_OUTPUT = os.getenv("SKIP_ON_BAD_OUTPUT", "1") == "1"
BAD_OUTPUT_MAX_CONTROL_CHAR_RATIO = float(os.getenv("BAD_OUTPUT_MAX_CONTROL_CHAR_RATIO", "0.2"))
BAD_OUTPUT_MAX_REPEATED_CHAR_RUN = int(os.getenv("BAD_OUTPUT_MAX_REPEATED_CHAR_RUN", "80"))
BAD_OUTPUT_MIN_LENGTH_FOR_CHECK = int(os.getenv("BAD_OUTPUT_MIN_LENGTH_FOR_CHECK", "80"))
MODEL_ID = "Qwen3.5 35B"

_GCS_CLIENT = None
FILE_SOURCE_MAP = {}

DEFAULT_SYSTEM_PROMPT = """
You are a strict MSc thesis evaluator.
Use internal reasoning in a <think>...</think> block, then output exactly one JSON object.

Score criteria and ranges:
- contribution: 0-25
- rigor: 0-20
- implementation: 0-20
- theory: 0-10
- process: 0-15
- impact: 0-10

Output schema:
{
  "scores": {"contribution": int, "rigor": int, "implementation": int, "theory": int, "process": int, "impact": int},
  "total": int,
  "delta_summary": "string",
  "verdict": "string"
}

Do not output markdown code fences.
""".strip()


def load_grading_preset(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        print(f"Preset file not found: {path}. Using script defaults.")
        return {}
    except Exception as exc:
        print(f"Failed loading preset {path}: {exc}. Using script defaults.")
        return {}

    fields = raw.get("operation", {}).get("fields", [])
    preset = {}
    for field in fields:
        key = field.get("key")
        if key:
            preset[key] = field.get("value")
    return preset


def load_research_questions(jsonl_path):
    """Load research questions from JSONL file into a dict keyed by filename.
    
    Returns:
        dict: Maps filename -> list of RQ strings. Missing files map to empty list.
    """
    rq_dict = {}
    if not os.path.exists(jsonl_path):
        print(f"Research questions file not found: {jsonl_path}. Proceeding without RQs.")
        return rq_dict

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    file_name = record.get("file_name", "").strip()
                    extraction = record.get("extraction", {})
                    rqs = extraction.get("research_questions", [])
                    if file_name and rqs:
                        rq_dict[file_name] = rqs
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Failed loading research questions from {jsonl_path}: {e}")

    print(f"Loaded research questions for {len(rq_dict)} theses.")
    return rq_dict


def format_research_questions_section(rqs):
    """Format research questions for inclusion in system prompt."""
    if not rqs or not isinstance(rqs, list):
        return ""
    
    rq_text = "\n\n### Research Questions This Thesis Must Address:\n"
    for i, rq in enumerate(rqs, 1):
        rq_text += f"{i}. {rq}\n"
    
    rq_text += "\nEvaluate how thoroughly the thesis addresses each of these research questions, especially when scoring research_question_alignment.\n"
    return rq_text


def build_response_format_from_preset(preset):
    structured_cfg = preset.get("llm.prediction.structured")
    if not isinstance(structured_cfg, dict):
        return None

    structured_type = str(structured_cfg.get("type", "")).strip().lower()
    if structured_type in ("", "none", "off", "disabled"):
        return None
    if structured_type not in ("json_schema", "jsonschema"):
        return None

    schema = structured_cfg.get("jsonSchema")
    if not isinstance(schema, dict):
        return None

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "thesis_grade_response",
            "strict": True,
            "schema": schema,
        },
    }


GRADING_PRESET = load_grading_preset(GRADING_PRESET_PATH)
RESEARCH_QUESTIONS_BY_FILE = load_research_questions(RESEARCH_QUESTIONS_JSONL)

def build_json_only_output_guard(response_format):
    """Create a strict output guard aligned with active JSON schema."""
    if not isinstance(response_format, dict):
        return ""
    if response_format.get("type") != "json_schema":
        return ""

    schema = response_format.get("json_schema", {}).get("schema", {})
    required = schema.get("required") or list((schema.get("properties") or {}).keys())
    required_keys_text = ", ".join(required)
    return (
        "\n\nCRITICAL OUTPUT RULES (override any conflicting instructions):\n"
        "- Return ONLY one JSON object that matches the schema.\n"
        "- Do NOT output markdown, tables, prose, explanations, or code fences.\n"
        f"- Required keys: {required_keys_text}."
    )


def process_streaming_response(stream):
    """Process OpenAI-compatible streaming output and capture optional reasoning deltas."""
    reasoning_buffer = []
    message_buffer = []
    reasoning_has_content = False
    event_count = {"openai": 0, "reasoning_deltas": 0, "message_deltas": 0}

    print("\n--- MODEL REASONING (streaming) ---")
    for event_data in stream:
        # OpenAI-compatible chunk shape
        event_count["openai"] += 1
        choices = getattr(event_data, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        # Standard delta content
        chunk_content = getattr(delta, "content", None)
        if chunk_content:
            event_count["message_deltas"] += 1
            message_buffer.append(chunk_content)

        # Optional reasoning content used by some models/providers
        reasoning_chunk = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if reasoning_chunk:
            reasoning_buffer.append(reasoning_chunk)
            reasoning_has_content = True
            print(reasoning_chunk, end="", flush=True)

    # Fallback: if message buffer is empty but reasoning was captured, use reasoning as content
    if not message_buffer and reasoning_has_content:
        message_buffer = reasoning_buffer
    
        print(f"\n[DEBUG: Event counts - OpenAI: {event_count['openai']}, "
            f"Reasoning deltas: {event_count['reasoning_deltas']}, Message deltas: {event_count['message_deltas']}]", flush=True)
    print(f"[DEBUG: Reasoning buffer size: {len(''.join(reasoning_buffer))} chars, "
          f"Message buffer size: {len(''.join(message_buffer))} chars]", flush=True)
    
    return "".join(reasoning_buffer), "".join(message_buffer)


def extract_text_from_pdf(path):
    doc = fitz.open(path)
    try:
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text() or "")
        text = "\n".join(text_parts)
        text_parts.clear()
        return text
    finally:
        doc.close()


def get_gcs_client():
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        if storage is None:
            raise RuntimeError(
                "google-cloud-storage is not installed. Install dependencies before using PDF_INPUT_SOURCE=gcs."
            )
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def extract_text_from_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text() or "")
        text = "\n".join(text_parts)
        text_parts.clear()
        return text
    finally:
        doc.close()


def list_pdf_filenames_from_gcs(bucket_name, prefix=""):
    """List PDF filenames from GCS and build FILE_SOURCE_MAP for on-demand download."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix or None)

    FILE_SOURCE_MAP.clear()
    duplicate_count = 0
    for blob in blobs:
        if not blob.name or blob.name.endswith("/"):
            continue
        if not blob.name.lower().endswith(".pdf"):
            continue

        filename = os.path.basename(blob.name)
        if filename in FILE_SOURCE_MAP:
            duplicate_count += 1
            print(
                f"Warning: Duplicate basename in GCS stream ({filename}). "
                f"Keeping first occurrence and skipping gs://{bucket_name}/{blob.name}."
            )
            continue

        FILE_SOURCE_MAP[filename] = {
            "type": "gcs",
            "bucket": bucket_name,
            "blob": blob.name,
        }

    if duplicate_count > 0:
        print(f"Skipped {duplicate_count} duplicate-basename blob(s) from GCS listing.")

    files = list(FILE_SOURCE_MAP.keys())
    files.sort()
    return files


def list_pdf_filenames_local(folder_path):
    """List local PDF filenames and build FILE_SOURCE_MAP for local reads."""
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    files.sort()

    FILE_SOURCE_MAP.clear()
    for filename in files:
        FILE_SOURCE_MAP[filename] = {
            "type": "local",
            "path": os.path.join(folder_path, filename),
        }
    return files


def extract_text_for_filename(filename):
    source_info = FILE_SOURCE_MAP.get(filename)
    if not source_info:
        # Fallback for legacy behavior when map has not been populated.
        return extract_text_from_pdf(os.path.join(PDF_FOLDER, filename))

    source_type = source_info.get("type")
    if source_type == "local":
        return extract_text_from_pdf(source_info["path"])

    if source_type == "gcs":
        client = get_gcs_client()
        bucket = client.bucket(source_info["bucket"])
        blob = bucket.blob(source_info["blob"])
        pdf_bytes = blob.download_as_bytes()
        try:
            return extract_text_from_pdf_bytes(pdf_bytes)
        finally:
            pdf_bytes = None

    raise RuntimeError(f"Unsupported PDF source type for {filename}: {source_type}")


def load_input_filenames():
    """Load candidate PDF filenames from configured input source."""
    if PDF_INPUT_SOURCE == "gcs":
        if not GCS_BUCKET_NAME:
            raise RuntimeError("GCS_BUCKET_NAME must be set when PDF_INPUT_SOURCE=gcs")
        print(
            f"Loading PDFs from GCS bucket {GCS_BUCKET_NAME} "
            f"with prefix '{GCS_PDF_PREFIX or '(none)'}'."
        )
        return list_pdf_filenames_from_gcs(GCS_BUCKET_NAME, GCS_PDF_PREFIX)

    if PDF_INPUT_SOURCE != "local":
        raise RuntimeError(f"Unsupported PDF_INPUT_SOURCE: {PDF_INPUT_SOURCE}. Use 'local' or 'gcs'.")

    print(f"Loading PDFs from local folder: {PDF_FOLDER}")
    return list_pdf_filenames_local(PDF_FOLDER)


def chars_to_tokens(char_count):
    return int(char_count / CHARS_PER_TOKEN_ESTIMATE)


def compact_text_for_prompt(text):
    """Reduce prompt token load while keeping paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def trim_text_before_appendix_or_references(text):
    """Trim thesis text at appendix/reference sections to reduce prompt size.

    Returns:
        (trimmed_text, did_trim, marker)
    """
    if not text:
        return text, False, ""

    heading_patterns = [
        r"(?im)^\s*appendix(?:es)?\b",
        r"(?im)^\s*references\b",
        r"(?im)^\s*bibliography\b",
        r"(?im)^\s*works\s+cited\b",
        r"(?im)^\s*litteratur(?:e|en)?\b",
        r"(?im)^\s*referencer\b",
    ]

    best_match = None
    best_start = None
    for pattern in heading_patterns:
        match = re.search(pattern, text)
        if match is None:
            continue
        start = match.start()
        if best_start is None or start < best_start:
            best_start = start
            best_match = match

    if best_match is None or best_start is None:
        return text, False, ""

    # Ignore pathological matches too early in the document.
    if best_start < 3000:
        return text, False, ""

    trimmed = text[:best_start].rstrip()
    if len(trimmed) >= len(text):
        return text, False, ""

    marker = best_match.group(0).strip()
    return trimmed, True, marker


def is_context_length_error(exc):
    """Detect provider errors indicating input exceeds model context."""
    error_text = str(exc).lower()
    return (
        "context length" in error_text
        or "maximum input length" in error_text
        or ("input tokens" in error_text and "requested" in error_text)
        or "parameter=input_tokens" in error_text
    )


def split_thinking_and_answer(content):
    """Extract thinking content and final answer. Handles Qwen3's behavior where chat template includes <think> automatically."""
    # Standard case: explicit <think>...</think> block
    think_match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if think_match:
        thinking_text = think_match.group(1).strip()
        answer_text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return thinking_text, answer_text

    # Qwen3 behavior: model returns only closing tag (implicit opening per chat template)
    # Everything before </think> is the thinking content
    if "</think>" in content:
        thinking_prefix, _, answer_suffix = content.partition("</think>")
        thinking_text = thinking_prefix.strip()
        answer_text = answer_suffix.strip()
        # Only treat prefix as thinking if it has substantial content
        if thinking_text and len(thinking_text) > 10:
            return thinking_text, answer_text
        else:
            return "", content.strip()

    return "", content.strip()


def extract_and_display_thinking(content):
    thinking_text, _ = split_thinking_and_answer(content)
    if thinking_text:
        print(f"\n--- MODEL THINKING ---\n{thinking_text}\n--- END THINKING ---\n")
    else:
        print("\n--- MODEL THINKING ---\n(No explicit <think> block returned.)\n--- END THINKING ---\n")


def extract_thinking_text(content, fallback_reasoning=""):
    """Extract best-available thinking text for logging."""
    thinking_text, _ = split_thinking_and_answer(content or "")
    if thinking_text:
        return thinking_text
    return (fallback_reasoning or "").strip()


def append_thinking_log(filename, thinking_text, meta=None):
    """Append model thinking to JSONL log when enabled."""
    if not LOG_MODEL_THINKING:
        return
    if not thinking_text:
        return

    if THINKING_MAX_CHARS > 0:
        thinking_text = thinking_text[:THINKING_MAX_CHARS]

    entry = {
        "file": filename,
        "thinking": thinking_text,
        "meta": meta or {},
        "timestamp": int(time.time()),
    }

    try:
        log_dir = os.path.dirname(THINKING_LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(THINKING_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Warning: Failed writing thinking log for {filename}: {exc}")


def parse_robust_json(content):
    _, content_clean = split_thinking_and_answer(content)
    content_clean = re.sub(r"```json|```", "", content_clean).strip()

    start = content_clean.find("{")
    end = content_clean.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        return json.loads(content_clean[start : end + 1])
    except json.JSONDecodeError:
        return None


def detect_bad_output_patterns(content):
    """Detect pathological model outputs caused by corrupted inputs.

    Returns:
        (is_bad, reason): Tuple[bool, str]
    """
    if not isinstance(content, str):
        return True, "non_string_output"

    text = content.strip()
    if len(text) < BAD_OUTPUT_MIN_LENGTH_FOR_CHECK:
        return False, ""

    # High density of known problematic zero-width/control-like Unicode marks.
    control_like = ["\u200e", "\u200f", "\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
    control_like_count = sum(text.count(ch) for ch in control_like)
    control_like_ratio = control_like_count / max(1, len(text))
    if control_like_ratio > BAD_OUTPUT_MAX_CONTROL_CHAR_RATIO:
        return True, f"control_char_ratio_exceeded:{control_like_ratio:.3f}"

    # Detect suspicious long runs of the same character.
    max_run = 1
    run = 1
    prev = None
    for ch in text:
        if ch == prev:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1
            prev = ch
    if max_run >= BAD_OUTPUT_MAX_REPEATED_CHAR_RUN:
        return True, f"repeated_char_run_exceeded:{max_run}"

    # If it fails JSON parse and has very low alphabetic signal, it is likely garbage.
    if parse_robust_json(text) is None:
        alpha_count = sum(1 for ch in text if ch.isalpha())
        alpha_ratio = alpha_count / max(1, len(text))
        if alpha_ratio < 0.05:
            return True, f"low_alpha_non_json_output:{alpha_ratio:.3f}"

    return False, ""


def load_already_graded(output_file):
    """Load set of successfully graded thesis filenames from output file."""
    graded = set()
    if not os.path.exists(output_file):
        return graded

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    # Only skip theses with parsed structured result.
                    if result.get("data") is not None:
                        graded.add(result.get("file"))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Warning: Failed to load graded files: {e}")

    return graded


def load_priority_filenames(csv_path, limit):
    """Load smallest-first PDF names from CSV with columns: rank, filename, character_count."""
    # Resolve common relative-path cases for robustness when cwd varies.
    if not os.path.isabs(csv_path):
        candidates = [
            csv_path,
            os.path.join(os.path.dirname(__file__), csv_path),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", csv_path)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", csv_path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                csv_path = candidate
                break

    if not os.path.exists(csv_path):
        print(f"Priority CSV not found: {csv_path}. Falling back to default file ordering.")
        return []

    ordered = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("filename") or "").strip()
                if name:
                    ordered.append(name)
                    if limit > 0 and len(ordered) >= limit:
                        break
    except Exception as exc:
        print(f"Failed loading priority CSV {csv_path}: {exc}. Falling back to default file ordering.")
        return []

    return ordered


def grade_thesis(filename, text):
    original_chars = len(text)
    print(f"Original text length: {original_chars} characters")

    if COMPACT_INPUT_WHITESPACE:
        text = compact_text_for_prompt(text)

    thesis_excerpt = text
    text = None

    system_prompt = GRADING_PRESET.get("llm.prediction.systemPrompt") or DEFAULT_SYSTEM_PROMPT
    request_temperature = float(GRADING_PRESET.get("llm.prediction.temperature", REQUEST_TEMPERATURE))
    request_top_p = REQUEST_TOP_P
    top_p_cfg = GRADING_PRESET.get("llm.prediction.topPSampling")
    if isinstance(top_p_cfg, dict) and top_p_cfg.get("checked"):
        request_top_p = float(top_p_cfg.get("value", REQUEST_TOP_P))

    response_format = build_response_format_from_preset(GRADING_PRESET)
    json_only_guard = build_json_only_output_guard(response_format)
    if json_only_guard:
        system_prompt = f"{system_prompt}\n{json_only_guard}"

    extra_body = {"top_k": 20}
    min_p_cfg = GRADING_PRESET.get("llm.prediction.minPSampling")
    if isinstance(min_p_cfg, dict) and min_p_cfg.get("checked"):
        extra_body["min_p"] = float(min_p_cfg.get("value", 0.0))

    last_error = None
    input_chars = len(thesis_excerpt)
    estimated_prompt_tokens = chars_to_tokens(input_chars)
    fit_attempts_used = 1
    trimmed_at_references = False
    trim_marker = ""
    was_truncated = False
    max_input_chars_effective = None
    response = None
    max_attempts_effective = MAX_RETRIES
    attempt = 0
    while attempt < max_attempts_effective:
        attempt += 1
        try:
            print(
                f"Prepared prompt: {len(thesis_excerpt)} chars "
                f"(~{estimated_prompt_tokens} tokens at {CHARS_PER_TOKEN_ESTIMATE:.1f} chars/token)"
            )

            request_kwargs = {
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Grade this thesis text.\n\n"
                            f"{thesis_excerpt}"
                        ),
                    },
                ],
                "temperature": request_temperature,
                "max_tokens": MAX_COMPLETION_TOKENS,
                "timeout": REQUEST_TIMEOUT_SECONDS,
            }
            if request_top_p is not None:
                request_kwargs["top_p"] = request_top_p
            if response_format is not None:
                request_kwargs["response_format"] = response_format
            if ENABLE_STREAM_THINKING:
                request_kwargs["stream"] = True

            if extra_body:
                request_kwargs["extra_body"] = extra_body

            reasoning_content = ""
            if ENABLE_STREAM_THINKING:
                response = client.chat.completions.create(**request_kwargs)
                reasoning_content, message_content = process_streaming_response(response)
                full_content = message_content or reasoning_content
                parse_content = message_content or full_content
            else:
                response = client.chat.completions.create(**request_kwargs)
                full_content = response.choices[0].message.content or ""
                parse_content = full_content

            if not full_content.strip() and not ENABLE_STREAM_THINKING and response is not None:
                parsed_payload = getattr(response.choices[0].message, "parsed", None)
                if parsed_payload is not None:
                    try:
                        full_content = json.dumps(parsed_payload, ensure_ascii=False)
                    except TypeError:
                        full_content = str(parsed_payload)
                    parse_content = full_content

            is_bad_output, bad_reason = detect_bad_output_patterns(full_content)
            if is_bad_output and SKIP_ON_BAD_OUTPUT:
                print(
                    f"Warning: Skipping {filename} due to suspicious output pattern ({bad_reason})."
                )
                return {
                    "file": filename,
                    "analysis": (full_content[:ANALYSIS_MAX_CHARS] if full_content else ""),
                    "data": None,
                    "meta": {
                        "attempts": attempt,
                        "original_chars": original_chars,
                        "trimmed_at_references": trimmed_at_references,
                        "trim_marker": trim_marker,
                        "input_chars": input_chars,
                        "estimated_input_tokens": estimated_prompt_tokens,
                        "max_input_chars_effective": max_input_chars_effective,
                        "was_truncated": was_truncated,
                        "prompt_fit_attempts": fit_attempts_used,
                        "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                        "stream_mode": "openai_compat_stream" if ENABLE_STREAM_THINKING else "non_stream",
                        "skipped_reason": "bad_output_detected",
                        "bad_output_detail": bad_reason,
                    },
                }

            thinking_text = extract_thinking_text(full_content, reasoning_content)
            if thinking_text:
                print(f"\n--- MODEL THINKING ---\n{thinking_text}\n--- END THINKING ---\n")
            else:
                print("\n--- MODEL THINKING ---\n(No explicit <think> block returned.)\n--- END THINKING ---\n")

            append_thinking_log(
                filename,
                thinking_text,
                {
                    "mode": "streaming" if ENABLE_STREAM_THINKING else "non_stream",
                    "attempt": attempt,
                },
            )

            structured_data = parse_robust_json(parse_content)
            if STORE_FULL_ANALYSIS:
                analysis_to_store = full_content
            else:
                analysis_to_store = full_content[:ANALYSIS_MAX_CHARS]

            response = None
            full_content = None
            thesis_excerpt = None

            return {
                "file": filename,
                "analysis": analysis_to_store,
                "data": structured_data,
                "meta": {
                    "attempts": attempt,
                    "original_chars": original_chars,
                    "trimmed_at_references": trimmed_at_references,
                    "trim_marker": trim_marker,
                    "input_chars": input_chars,
                    "estimated_input_tokens": estimated_prompt_tokens,
                    "max_input_chars_effective": max_input_chars_effective,
                    "was_truncated": was_truncated,
                    "prompt_fit_attempts": fit_attempts_used,
                    "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                    "stream_mode": "openai_compat_stream" if ENABLE_STREAM_THINKING else "non_stream",
                },
            }
        except Exception as exc:
            last_error = exc

            # On context overflow, trim appendix/references once and retry immediately.
            if is_context_length_error(exc) and not trimmed_at_references:
                trimmed_text, did_trim, marker = trim_text_before_appendix_or_references(thesis_excerpt)
                if did_trim:
                    old_chars = len(thesis_excerpt)
                    thesis_excerpt = trimmed_text
                    input_chars = len(thesis_excerpt)
                    estimated_prompt_tokens = chars_to_tokens(input_chars)
                    trimmed_at_references = True
                    trim_marker = marker
                    was_truncated = True
                    max_input_chars_effective = input_chars
                    fit_attempts_used += 1
                    max_attempts_effective += 1
                    print(
                        f"Context overflow for {filename}. Trimmed at heading '{trim_marker}' "
                        f"({old_chars} -> {input_chars} chars) and retrying."
                    )
                    continue

            if attempt < max_attempts_effective:
                sleep_seconds = min(12, 2 ** attempt + random.random())
                print(
                    f"Retry {attempt}/{max_attempts_effective} after error: {exc} "
                    f"(sleep {sleep_seconds:.1f}s)"
                )
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Model request failed after {MAX_RETRIES} attempts: {last_error}")


def build_batch_requests(filenames, rq_dict=None):
    """Build a list of batch API request dicts for all theses.
    
    Args:
        filenames: List of PDF filenames to grade.
        rq_dict: Optional dict mapping filename -> list of research questions.
    """
    if rq_dict is None:
        rq_dict = {}
    
    # Get preset configs once
    system_prompt_base = GRADING_PRESET.get("llm.prediction.systemPrompt") or DEFAULT_SYSTEM_PROMPT
    request_temperature = float(GRADING_PRESET.get("llm.prediction.temperature", REQUEST_TEMPERATURE))
    request_top_p = REQUEST_TOP_P
    top_p_cfg = GRADING_PRESET.get("llm.prediction.topPSampling")
    if isinstance(top_p_cfg, dict) and top_p_cfg.get("checked"):
        request_top_p = float(top_p_cfg.get("value", REQUEST_TOP_P))

    response_format = build_response_format_from_preset(GRADING_PRESET)
    json_only_guard = build_json_only_output_guard(response_format)

    extra_body = {"top_k": 20}
    min_p_cfg = GRADING_PRESET.get("llm.prediction.minPSampling")
    if isinstance(min_p_cfg, dict) and min_p_cfg.get("checked"):
        extra_body["min_p"] = float(min_p_cfg.get("value", 0.0))

    requests = []
    selected_filenames = []
    skipped_already_graded = 0
    already_graded = load_already_graded(OUTPUT_FILE)
    total_files = len(filenames)
    started_at = time.time()
    progress_every = 1 if total_files <= 20 else max(5, total_files // 20)

    for idx, filename in enumerate(filenames, start=1):
        if filename in already_graded:
            skipped_already_graded += 1
            if idx == 1 or idx == total_files or idx % progress_every == 0:
                elapsed = time.time() - started_at
                pct = (idx / total_files) * 100 if total_files else 100.0
                print(
                    f"Built batch requests: {idx}/{total_files} ({pct:.1f}%) "
                    f"elapsed {elapsed:.1f}s - skipped already-graded {filename}"
                )
            continue

        text = extract_text_for_filename(filename)
        
        if COMPACT_INPUT_WHITESPACE:
            text = compact_text_for_prompt(text)

        # Build system prompt with research questions
        system_prompt = system_prompt_base
        rqs_section = format_research_questions_section(rq_dict.get(filename, []))
        if rqs_section:
            system_prompt = f"{system_prompt}\n{rqs_section}"
        
        # Add JSON output guard
        if json_only_guard:
            system_prompt = f"{system_prompt}\n{json_only_guard}"

        body = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Grade this thesis text.\n\n{text}",
                },
            ],
            "temperature": request_temperature,
            "max_tokens": MAX_COMPLETION_TOKENS,
        }
        if request_top_p is not None:
            body["top_p"] = request_top_p
        if response_format is not None:
            body["response_format"] = response_format
        if extra_body:
            body["extra_body"] = extra_body

        requests.append({
            "custom_id": filename,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })
        selected_filenames.append(filename)

        if idx == 1 or idx == total_files or idx % progress_every == 0:
            elapsed = time.time() - started_at
            pct = (idx / total_files) * 100 if total_files else 100.0
            print(
                f"Built batch requests: {idx}/{total_files} ({pct:.1f}%) "
                f"elapsed {elapsed:.1f}s - {filename}"
            )
        
        text = None  # Clear for memory efficiency

    return requests, selected_filenames, skipped_already_graded


def _extract_file_content_text(file_content_obj):
    """Normalize Files API content payloads across SDK/provider variants."""
    text_attr = getattr(file_content_obj, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    content_attr = getattr(file_content_obj, "content", None)
    if isinstance(content_attr, bytes):
        return content_attr.decode("utf-8", errors="replace")
    if isinstance(content_attr, str):
        return content_attr

    return str(file_content_obj)


def _get_batch_counts(batch_status):
    """Return completed/total counts from request_counts dict or object."""
    request_counts = getattr(batch_status, "request_counts", None)
    if request_counts is None:
        return None, None

    if isinstance(request_counts, dict):
        return request_counts.get("completed"), request_counts.get("total")

    return getattr(request_counts, "completed", None), getattr(request_counts, "total", None)


def submit_grading_batch(filenames):
    """Submit all theses for grading via OpenAI Batch API and wait for completion."""
    print(f"\n=== BATCH MODE: Submitting {len(filenames)} theses ===")
    
    # Build requests
    print(f"Building batch requests for {len(filenames)} files...")
    requests, selected_filenames, skipped_already_graded = build_batch_requests(
        filenames,
        rq_dict=RESEARCH_QUESTIONS_BY_FILE,
    )

    if skipped_already_graded > 0:
        print(f"Skipped {skipped_already_graded} already-graded file(s) during batch preparation.")

    if not requests:
        print("No new files to submit in batch (all selected files already graded).")
        return "skipped_all_already_graded"
    
    # Write to temp JSONL file
    batch_jsonl_path = "batch_input.jsonl"
    print(f"Writing {len(requests)} requests to {batch_jsonl_path}...")
    with open(batch_jsonl_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    
    # Upload file
    print("Uploading batch file to OpenAI Files API...")
    try:
        with open(batch_jsonl_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="batch")
    except Exception as exc:
        print(f"Batch upload failed: {exc}")
        print(
            "Batch API may be unsupported on this endpoint (Files API disabled). "
            "Falling back to non-batch mode."
        )
        return None
    
    input_file_id = file_response.id
    print(f"File uploaded: {input_file_id}")
    
    # Create batch
    print("Creating batch job...")
    batch_response = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    
    batch_id = batch_response.id
    print(f"Batch created: {batch_id}")
    print(f"Status: {batch_response.status}")
    
    # Poll for completion
    poll_interval = 30
    max_wait_seconds = 86400  # 24 hours
    elapsed = 0
    
    print(f"Polling every {poll_interval}s for completion (timeout: {max_wait_seconds}s)...")
    while elapsed < max_wait_seconds:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status

        completed_count, total_count = _get_batch_counts(batch_status)
        if completed_count is not None and total_count is not None:
            print(f"[{elapsed}s] Status: {status} | Processed: {completed_count}/{total_count}")
        else:
            print(f"[{elapsed}s] Status: {status}")
        
        if status == "completed":
            print("✓ Batch completed!")
            break
        elif status in ["failed", "expired", "cancelled"]:
            print(f"✗ Batch {status}!")
            errors_file_id = getattr(batch_status, "errors_file_id", None)
            if errors_file_id:
                print("Downloading error file...")
                error_file_obj = client.files.content(errors_file_id)
                error_content = _extract_file_content_text(error_file_obj)
                print("--- BATCH ERRORS ---")
                print(error_content[:2000])
                print("--- END ERRORS ---")
            return None
        
        time.sleep(poll_interval)
        elapsed += poll_interval
    
    if elapsed >= max_wait_seconds:
        print("✗ Batch timed out!")
        return None
    
    # Download results
    print(f"Downloading results file {batch_status.output_file_id}...")
    output_file_obj = client.files.content(batch_status.output_file_id)
    output_content = _extract_file_content_text(output_file_obj)
    
    # Parse results and merge into grading_results.jsonl
    print("Parsing and saving results...")
    results_by_filename = {}
    for line in output_content.strip().split("\n"):
        if not line.strip():
            continue
        custom_id = None
        try:
            batch_result = json.loads(line)
            custom_id = batch_result.get("custom_id")
            response_body = batch_result.get("response", {}).get("body", {})
            
            # Extract message content
            message_content = ""
            choices = response_body.get("choices", [])
            if choices:
                message_content = choices[0].get("message", {}).get("content", "")
            
            # Parse structured data
            structured_data = parse_robust_json(message_content)
            thinking_text = extract_thinking_text(message_content)
            append_thinking_log(
                custom_id,
                thinking_text,
                {
                    "mode": "batch",
                    "batch_id": batch_id,
                },
            )
            
            result_entry = {
                "file": custom_id,
                "analysis": message_content[:ANALYSIS_MAX_CHARS] if message_content else "",
                "data": structured_data,
                "meta": {
                    "batch_mode": True,
                    "batch_id": batch_id,
                },
            }
            
            results_by_filename[custom_id] = result_entry
        except Exception as e:
            print(f"Warning: Failed to parse result for {custom_id}: {e}")
    
    # Append to output file
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for filename in selected_filenames:
            if filename in results_by_filename:
                f.write(json.dumps(results_by_filename[filename], ensure_ascii=False) + "\n")
    
    # Cleanup temp file
    try:
        os.remove(batch_jsonl_path)
    except Exception:
        pass
    
    print(f"✓ Batch complete! Saved {len(results_by_filename)} results to {OUTPUT_FILE}")
    return batch_id


def process_non_batch(filenames):
    """Process theses one-by-one via direct chat completions."""
    print(f"\n=== NON-BATCH MODE: Processing {len(filenames)} theses ===")
    completed = 0
    skipped_already_graded = 0
    total = len(filenames)
    already_graded = load_already_graded(OUTPUT_FILE)

    for idx, filename in enumerate(filenames, start=1):
        disk_graded = load_already_graded(OUTPUT_FILE)
        if disk_graded:
            already_graded.update(disk_graded)

        if filename in already_graded:
            skipped_already_graded += 1
            print(f"[{idx}/{total}] Skipping already-graded file: {filename}")
            continue

        print(f"[{idx}/{total}] Grading {filename}")
        try:
            text = extract_text_for_filename(filename)
            result = grade_thesis(filename, text)
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            completed += 1
            if result.get("data") is not None:
                already_graded.add(filename)
        except Exception as exc:
            print(f"Failed grading {filename}: {exc}")

        if PAUSE_BETWEEN_FILES_SECONDS > 0 and idx < total:
            time.sleep(PAUSE_BETWEEN_FILES_SECONDS)

    print(
        f"✓ Non-batch complete! Saved {completed}/{total} results to {OUTPUT_FILE} "
        f"(skipped already-graded: {skipped_already_graded})"
    )


def display_system_prompt(filename=None):
    """Display the complete system prompt that will be sent to the LLM.
    
    Args:
        filename: Optional PDF filename. If provided, includes research questions for that file.
                  If None, shows just the base system prompt with JSON guard.
    """
    # Get base system prompt from preset or default
    system_prompt = GRADING_PRESET.get("llm.prediction.systemPrompt") or DEFAULT_SYSTEM_PROMPT
    
    # Add research questions if filename is provided
    if filename:
        rqs_section = format_research_questions_section(RESEARCH_QUESTIONS_BY_FILE.get(filename, []))
        if rqs_section:
            system_prompt = f"{system_prompt}\n{rqs_section}"
    
    # Add JSON output guard
    response_format = build_response_format_from_preset(GRADING_PRESET)
    json_only_guard = build_json_only_output_guard(response_format)
    if json_only_guard:
        system_prompt = f"{system_prompt}\n{json_only_guard}"
    
    # Print the complete prompt
    print("=" * 80)
    print("COMPLETE SYSTEM PROMPT TO BE SENT TO LLM")
    print("=" * 80)
    if filename:
        print(f"File: {filename}")
    print()
    print(system_prompt)
    print()
    print("=" * 80)
    print(f"Total characters: {len(system_prompt)}")
    print(f"Estimated tokens: {chars_to_tokens(len(system_prompt))}")
    print("=" * 80)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Grade MSc theses with streaming or batch mode.")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch API mode for this run (default is streaming/non-batch).",
    )
    parser.add_argument(
        "--system_prompt",
        nargs="?",
        const="",
        metavar="FILENAME",
        help="Display the system prompt that will be sent to the LLM. Optionally provide a PDF filename to include its research questions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    
    # Handle --system_prompt flag
    if args.system_prompt is not None:
        # args.system_prompt will be "" if flag is provided without filename, or the filename if provided
        filename = args.system_prompt.strip() if args.system_prompt else None
        display_system_prompt(filename)
        sys.exit(0)
    
    use_batch_api = USE_BATCH_API or args.batch

    files = load_input_filenames()
    
    # Informational startup summary; final skip decision is done per file right before API submission.
    already_graded = load_already_graded(OUTPUT_FILE)
    if len(already_graded) > 0:
        print(
            f"Detected {len(already_graded)} previously graded files in {OUTPUT_FILE}. "
            "Each file will still be re-checked immediately before LLM submission."
        )

    if PRIORITIZE_SMALLEST_FROM_CSV and PDF_INPUT_SOURCE != "gcs":
        prioritized = load_priority_filenames(PDF_CHAR_COUNT_CSV, PRIORITY_FILE_LIMIT)
        if prioritized:
            slice_start = max(0, PRIORITY_START_INDEX)
            slice_end = PRIORITY_END_INDEX if PRIORITY_END_INDEX > 0 else None
            if slice_end is not None and slice_end < slice_start:
                print(
                    f"Invalid priority slice: start_index={slice_start}, end_index={slice_end}. "
                    "Resulting selection is empty."
                )
                prioritized = []
            else:
                prioritized = prioritized[slice_start:slice_end]
                if slice_end is None:
                    print(f"Using CSV priority slice from index {slice_start} to end.")
                else:
                    print(f"Using CSV priority slice [{slice_start}:{slice_end}].")
            available = set(files)
            files = [name for name in prioritized if name in available]
            print(
                f"Prioritized smallest files from CSV: selected {len(files)} file(s) "
                f"(limit={PRIORITY_FILE_LIMIT}, start_index={PRIORITY_START_INDEX}, "
                f"end_index={PRIORITY_END_INDEX}, csv={PDF_CHAR_COUNT_CSV})."
            )
    elif PRIORITIZE_SMALLEST_FROM_CSV and PDF_INPUT_SOURCE == "gcs":
        print("CSV priority ordering is disabled in GCS input mode.")
    
    if CANARY_MAX_FILES > 0:
        files = files[:CANARY_MAX_FILES]

    if not files:
        print("No files to process.")
        sys.exit(0)

    if use_batch_api:
        print("Run mode: BATCH (enabled via env USE_BATCH_API=1 or --batch)")
        batch_id = submit_grading_batch(files)
        if batch_id is None:
            process_non_batch(files)
    else:
        print("Run mode: STREAMING/NON-BATCH (default)")
        process_non_batch(files)
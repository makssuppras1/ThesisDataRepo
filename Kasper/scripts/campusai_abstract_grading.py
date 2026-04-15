import argparse
import gc
import json
import os
import random
import re
import sys
import time

import pymupdf as fitz
from openai import OpenAI

try:
    from google.cloud import storage
except ImportError:
    storage = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
    script_env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(script_env_path, override=False)

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "4000"))
CHARS_PER_TOKEN_ESTIMATE = float(os.getenv("CHARS_PER_TOKEN_ESTIMATE", "3.3"))
REQUEST_TEMPERATURE = float(os.getenv("REQUEST_TEMPERATURE", "0.7"))
REQUEST_TOP_P = float(os.getenv("REQUEST_TOP_P", "0.95"))
COMPACT_INPUT_WHITESPACE = os.getenv("COMPACT_INPUT_WHITESPACE", "0") == "0"
CANARY_MAX_FILES = int(os.getenv("CANARY_MAX_FILES", "0"))
STORE_FULL_ANALYSIS = os.getenv("STORE_FULL_ANALYSIS", "0") == "1"
ANALYSIS_MAX_CHARS = int(os.getenv("ANALYSIS_MAX_CHARS", "8000"))
FORCE_GC_EACH_FILE = os.getenv("FORCE_GC_EACH_FILE", "0") == "1"
PAUSE_BETWEEN_FILES_SECONDS = float(os.getenv("PAUSE_BETWEEN_FILES_SECONDS", "0"))
PRIORITIZE_SMALLEST_FROM_CSV = os.getenv("PRIORITIZE_SMALLEST_FROM_CSV", "1") == "1"
PRIORITY_FILE_LIMIT = int(os.getenv("PRIORITY_FILE_LIMIT", "0"))
PRIORITY_START_INDEX = int(os.getenv("PRIORITY_START_INDEX", "0"))
PRIORITY_END_INDEX = int(os.getenv("PRIORITY_END_INDEX", "2000"))
PDF_CHAR_COUNT_CSV = os.getenv(
    "PDF_CHAR_COUNT_CSV",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pdf_char_count_sorted.csv")),
)
ABSTRACT_PRESET_PATH = os.getenv(
    "ABSTRACT_PRESET_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "abstract_grading.preset.json")),
)
ENABLE_STREAM_THINKING = os.getenv("ENABLE_STREAM_THINKING", "1") == "1"
API_BASE_URL = os.getenv("OPENAI_BASE_URL", os.getenv("CAMPUSAI_BASE_URL", "https://api.campusai.compute.dtu.dk/v1"))
API_KEY = os.getenv("CAMPUSAI_API_KEY") or os.getenv("OPENAI_API_KEY")
PDF_FOLDER = r"C:\Users\kkrus\OneDrive - Danmarks Tekniske Universitet\DTU\MScProject\thesis_pdfs"
PDF_INPUT_SOURCE = os.getenv("PDF_INPUT_SOURCE", "gcs").strip().lower()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "thesis_archive_bucket").strip()
GCS_PDF_PREFIX = os.getenv("GCS_PDF_PREFIX", "dtu_findit/master_thesis").strip()
OUTPUT_FILE = os.getenv(
    "ABSTRACT_OUTPUT_FILE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "abstract_grading_results.jsonl")),
)
THINKING_LOG_FILE = os.getenv(
    "THINKING_LOG_FILE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "abstract_grading_thinking.jsonl")),
)
LOG_MODEL_THINKING = os.getenv("LOG_MODEL_THINKING", "1") == "1"
THINKING_MAX_CHARS = int(os.getenv("THINKING_MAX_CHARS", "40000"))
SKIP_ON_BAD_OUTPUT = os.getenv("SKIP_ON_BAD_OUTPUT", "1") == "1"
BAD_OUTPUT_MAX_CONTROL_CHAR_RATIO = float(os.getenv("BAD_OUTPUT_MAX_CONTROL_CHAR_RATIO", "0.2"))
BAD_OUTPUT_MAX_REPEATED_CHAR_RUN = int(os.getenv("BAD_OUTPUT_MAX_REPEATED_CHAR_RUN", "80"))
BAD_OUTPUT_MIN_LENGTH_FOR_CHECK = int(os.getenv("BAD_OUTPUT_MIN_LENGTH_FOR_CHECK", "80"))
MODEL_ID = os.getenv("MODEL_ID", "Qwen3.5 35B")

_GCS_CLIENT = None
FILE_SOURCE_MAP = {}

ABSTRACT_SCORE_KEYS = [
    "scope_and_clarity",
    "methodological_signal",
    "results_and_evidence_signal",
    "impact_and_alignment",
]

DEFAULT_SYSTEM_PROMPT = """
You are a strict MSc thesis abstract evaluator.
Use internal reasoning in a <think>...</think> block, then output exactly one JSON object.
You only see the abstract, so judge conservatively and do not invent missing thesis details.

Important context for abstracts:
- A thesis abstract is typically a short public-facing summary (often around 250 words).
- It does not need a rigid formal structure or fixed elements.
- Do not penalize missing headings or strict section ordering.
- Score what is communicated clearly, not whether a template was followed.

Score criteria and ranges:
- scope_and_clarity: 0-25
- methodological_signal: 0-25
- results_and_evidence_signal: 0-25
- impact_and_alignment: 0-25

Output schema:
{
  "scope_and_clarity": int,
  "methodological_signal": int,
  "results_and_evidence_signal": int,
  "impact_and_alignment": int,
  "total_score": int
}

The total_score must be on a 0-100 scale and should reflect the four criterion scores scaled to that range.
Do not output markdown code fences.
""".strip()


def load_abstract_preset(path):
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
            "name": "abstract_grade_response",
            "strict": True,
            "schema": schema,
        },
    }


ABSTRACT_GRADING_PRESET = load_abstract_preset(ABSTRACT_PRESET_PATH)


def build_json_only_output_guard(response_format):
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


if not API_KEY:
    raise RuntimeError(
        "Missing API key. Set OPENAI_API_KEY or CAMPUSAI_API_KEY in environment or .env file."
    )

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
    timeout=REQUEST_TIMEOUT_SECONDS,
)


def get_gcs_client():
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        if storage is None:
            raise RuntimeError(
                "google-cloud-storage is not installed. Install dependencies before using PDF_INPUT_SOURCE=gcs."
            )
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def extract_text_from_pdf(path):
    doc = fitz.open(path)
    try:
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text() or "")
        return "\n".join(text_parts)
    finally:
        doc.close()


def extract_text_from_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text() or "")
        return "\n".join(text_parts)
    finally:
        doc.close()


def list_pdf_filenames_from_gcs(bucket_name, prefix=""):
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
    if PDF_INPUT_SOURCE == "gcs":
        if not GCS_BUCKET_NAME:
            raise RuntimeError("GCS_BUCKET_NAME must be set when PDF_INPUT_SOURCE=gcs")
        print(
            f"Loading PDFs from GCS bucket {GCS_BUCKET_NAME} with prefix '{GCS_PDF_PREFIX or '(none)'}'."
        )
        return list_pdf_filenames_from_gcs(GCS_BUCKET_NAME, GCS_PDF_PREFIX)

    if PDF_INPUT_SOURCE != "local":
        raise RuntimeError(f"Unsupported PDF_INPUT_SOURCE: {PDF_INPUT_SOURCE}. Use 'local' or 'gcs'.")

    print(f"Loading PDFs from local folder: {PDF_FOLDER}")
    return list_pdf_filenames_local(PDF_FOLDER)


def load_already_graded(output_file):
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
                    if result.get("data") is not None:
                        graded.add(result.get("file"))
                except json.JSONDecodeError:
                    pass
    except Exception as exc:
        print(f"Warning: Failed to load graded files: {exc}")

    return graded


def load_priority_filenames(csv_path, limit):
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
        import csv

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


def compact_text_for_prompt(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_abstract_from_text(text):
    if not text:
        return "", False, ""

    control_re = re.compile(r'(?mi)^[\x00-\x1F\s\d\-\u00A0]*abstract\s*:?\s*$', re.MULTILINE)
    match = control_re.search(text)
    if match:
        start_pos = match.end()
        remainder = text[start_pos:]
        next_heading = re.search(r'(?m)^(?:#{1,6}\s|[A-Z][A-Z0-9 ,:/&()\'".\-]{3,}\s*$|\d+(?:\.\d+)*\s+[A-Z].*$|(^.+\n[-=]{3,}\s*$))', remainder)
        end_pos = next_heading.start() if next_heading else len(remainder)
        snippet = remainder[:end_pos]
        out_lines = snippet.splitlines()
        while out_lines and out_lines[0].strip() == "":
            out_lines.pop(0)
        while out_lines and out_lines[-1].strip() == "":
            out_lines.pop()
        abstract = "\n".join(out_lines).strip()
        return abstract, bool(abstract), "abstract"

    atx_re = re.compile(r'(?mi)^(#{1,6})\s*(abstract)\s*:?\s*$', re.MULTILINE)
    match = atx_re.search(text)
    if match:
        level = len(match.group(1))
        start_pos = match.end()
        lines = text[start_pos:].splitlines()
        out_lines = []
        stop_re = re.compile(r'^\s#{1,%d}\s' % level)
        for line in lines:
            if stop_re.match(line):
                break
            out_lines.append(line)
        while out_lines and out_lines[0].strip() == "":
            out_lines.pop(0)
        while out_lines and out_lines[-1].strip() == "":
            out_lines.pop()
        abstract = "\n".join(out_lines).strip()
        return abstract, bool(abstract), "abstract"

    setext_re = re.compile(r'(?mi)^(abstract)\s*\r?\n([=-]{3,})', re.MULTILINE)
    match = setext_re.search(text)
    if match:
        start_pos = match.end()
        lines = text[start_pos:].splitlines()
        out_lines = []
        stop_re = re.compile(r'^\s#{1,6}\s')
        for line in lines:
            if stop_re.match(line):
                break
            out_lines.append(line)
        while out_lines and out_lines[0].strip() == "":
            out_lines.pop(0)
        while out_lines and out_lines[-1].strip() == "":
            out_lines.pop()
        abstract = "\n".join(out_lines).strip()
        return abstract, bool(abstract), "abstract"

    return "", False, ""


def extract_abstract_for_filename(filename):
    text = extract_text_for_filename(filename)
    abstract, found, marker = extract_abstract_from_text(text)
    return abstract, found, marker, len(text)


def chars_to_tokens(char_count):
    return int(char_count / CHARS_PER_TOKEN_ESTIMATE)


def parse_robust_json(content):
    _, content_clean = split_thinking_and_answer(content or "")
    content_clean = re.sub(r"```json|```", "", content_clean).strip()
    start = content_clean.find("{")
    end = content_clean.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        return json.loads(content_clean[start : end + 1])
    except json.JSONDecodeError:
        return None


def extract_structured_payload(message):
    if message is None:
        return None

    parsed_payload = getattr(message, "parsed", None)
    if parsed_payload is not None:
        if isinstance(parsed_payload, dict):
            return parsed_payload
        try:
            return parsed_payload.model_dump()
        except Exception:
            try:
                return dict(parsed_payload)
            except Exception:
                return parsed_payload

    message_content = getattr(message, "content", None) or ""
    if message_content:
        return parse_robust_json(message_content)

    return None


def detect_bad_output_patterns(content):
    if not isinstance(content, str):
        return True, "non_string_output"

    text = content.strip()
    if len(text) < BAD_OUTPUT_MIN_LENGTH_FOR_CHECK:
        return False, ""

    control_like = ["\u200e", "\u200f", "\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
    control_like_count = sum(text.count(ch) for ch in control_like)
    control_like_ratio = control_like_count / max(1, len(text))
    if control_like_ratio > BAD_OUTPUT_MAX_CONTROL_CHAR_RATIO:
        return True, f"control_char_ratio_exceeded:{control_like_ratio:.3f}"

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

    if parse_robust_json(text) is None:
        alpha_count = sum(1 for ch in text if ch.isalpha())
        alpha_ratio = alpha_count / max(1, len(text))
        if alpha_ratio < 0.05:
            return True, f"low_alpha_non_json_output:{alpha_ratio:.3f}"

    return False, ""


def split_thinking_and_answer(content):
    think_match = re.search(r"<think>(.*?)</think>", content or "", flags=re.DOTALL)
    if think_match:
        thinking_text = think_match.group(1).strip()
        answer_text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
        return thinking_text, answer_text

    if "</think>" in (content or ""):
        thinking_prefix, _, answer_suffix = content.partition("</think>")
        thinking_text = thinking_prefix.strip()
        answer_text = answer_suffix.strip()
        if thinking_text and len(thinking_text) > 10:
            return thinking_text, answer_text
        return "", (content or "").strip()

    return "", (content or "").strip()


def extract_thinking_text(content, fallback_reasoning=""):
    thinking_text, _ = split_thinking_and_answer(content or "")
    if thinking_text:
        return thinking_text
    return (fallback_reasoning or "").strip()


def append_thinking_log(filename, thinking_text, meta=None):
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


def normalize_abstract_scores(data):
    if not isinstance(data, dict):
        return None

    values = []
    for key in ABSTRACT_SCORE_KEYS:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return data
        values.append(value)

    data["total_score"] = int(sum(values))
    return data


def build_result_record(filename, structured_data, abstract_found=True, abstract_marker="abstract"):
    """Build the minimal persisted result payload for JSONL output."""
    return {
        "file": filename,
        "data": structured_data,
        "abstract_found": bool(abstract_found),
        "abstract_marker": abstract_marker or "",
    }


def grade_abstract(filename, abstract_text, original_chars, abstract_found=True, abstract_marker=""):
    if COMPACT_INPUT_WHITESPACE:
        abstract_text = compact_text_for_prompt(abstract_text)

    abstract_excerpt = abstract_text
    abstract_text = None

    system_prompt = ABSTRACT_GRADING_PRESET.get("llm.prediction.systemPrompt") or DEFAULT_SYSTEM_PROMPT
    request_temperature = float(ABSTRACT_GRADING_PRESET.get("llm.prediction.temperature", REQUEST_TEMPERATURE))
    request_top_p = REQUEST_TOP_P
    top_p_cfg = ABSTRACT_GRADING_PRESET.get("llm.prediction.topPSampling")
    if isinstance(top_p_cfg, dict) and top_p_cfg.get("checked"):
        request_top_p = float(top_p_cfg.get("value", REQUEST_TOP_P))

    response_format = build_response_format_from_preset(ABSTRACT_GRADING_PRESET)
    json_only_guard = build_json_only_output_guard(response_format)
    if json_only_guard:
        system_prompt = f"{system_prompt}\n{json_only_guard}"

    extra_body = {"top_k": 20}
    min_p_cfg = ABSTRACT_GRADING_PRESET.get("llm.prediction.minPSampling")
    if isinstance(min_p_cfg, dict) and min_p_cfg.get("checked"):
        extra_body["min_p"] = float(min_p_cfg.get("value", 0.0))

    last_error = None
    input_chars = len(abstract_excerpt)
    estimated_prompt_tokens = chars_to_tokens(input_chars)
    response = None
    attempt = 0

    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            print(
                f"Prepared abstract prompt: {len(abstract_excerpt)} chars "
                f"(~{estimated_prompt_tokens} tokens at {CHARS_PER_TOKEN_ESTIMATE:.1f} chars/token)"
            )

            request_kwargs = {
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Grade this thesis abstract only. Do not assume details that are not stated.\n\n"
                            f"{abstract_excerpt}"
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
            message_content = ""
            if ENABLE_STREAM_THINKING:
                response = client.chat.completions.create(**request_kwargs)
                reasoning_buffer = []
                message_buffer = []
                print("\n--- MODEL REASONING (streaming) ---")
                for event_data in response:
                    choices = getattr(event_data, "choices", None)
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    if delta is None:
                        continue

                    chunk_content = getattr(delta, "content", None)
                    if chunk_content:
                        message_buffer.append(chunk_content)

                    reasoning_chunk = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                    if reasoning_chunk:
                        reasoning_buffer.append(reasoning_chunk)
                        print(reasoning_chunk, end="", flush=True)

                reasoning_content = "".join(reasoning_buffer)
                message_content = "".join(message_buffer)
                full_content = message_content or reasoning_content
                parse_content = message_content or full_content
                print(
                    f"\n[DEBUG: Reasoning buffer size: {len(reasoning_content)} chars, Message buffer size: {len(message_content)} chars]",
                    flush=True,
                )
            else:
                response = client.chat.completions.create(**request_kwargs)
                message = response.choices[0].message
                full_content = message.content or ""
                parse_content = full_content
                parsed_payload = extract_structured_payload(message)
                if parsed_payload is not None:
                    try:
                        full_content = json.dumps(parsed_payload, ensure_ascii=False)
                    except TypeError:
                        full_content = str(parsed_payload)
                    parse_content = full_content

            is_bad_output, bad_reason = detect_bad_output_patterns(full_content)
            if is_bad_output and SKIP_ON_BAD_OUTPUT:
                print(f"Warning: Skipping {filename} due to suspicious output pattern ({bad_reason}).")
                return build_result_record(
                    filename,
                    None,
                    abstract_found=abstract_found,
                    abstract_marker=abstract_marker,
                )

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
                    "abstract_only": True,
                },
            )

            structured_data = parse_robust_json(parse_content)

            if structured_data is None and ENABLE_STREAM_THINKING:
                # Some providers emit only reasoning deltas in stream mode.
                # Retry once in non-stream mode to force a final message payload.
                print(
                    f"Info: No parseable JSON in streaming response for {filename}. "
                    "Retrying once in non-stream mode."
                )
                fallback_kwargs = dict(request_kwargs)
                fallback_kwargs.pop("stream", None)
                fallback_kwargs["messages"] = list(request_kwargs["messages"]) + [
                    {
                        "role": "user",
                        "content": (
                            "Return ONLY the final JSON object now. "
                            "No additional text, no markdown, no explanation."
                        ),
                    }
                ]
                fallback_response = client.chat.completions.create(**fallback_kwargs)
                fallback_message = fallback_response.choices[0].message
                fallback_content = fallback_message.content or ""
                parsed_payload = extract_structured_payload(fallback_message)
                if parsed_payload is not None:
                    try:
                        fallback_content = json.dumps(parsed_payload, ensure_ascii=False)
                    except TypeError:
                        fallback_content = str(parsed_payload)
                if fallback_content:
                    full_content = fallback_content
                    parse_content = fallback_content
                structured_data = parse_robust_json(parse_content)
                if structured_data is None:
                    print(
                        f"Warning: No parseable JSON after non-stream fallback for {filename}."
                    )

            structured_data = normalize_abstract_scores(structured_data)

            response = None
            full_content = None
            abstract_excerpt = None

            return build_result_record(
                filename,
                structured_data,
                abstract_found=abstract_found,
                abstract_marker=abstract_marker,
            )
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                sleep_seconds = min(12, 2 ** attempt + random.random())
                print(f"Retry {attempt}/{MAX_RETRIES} after error: {exc} (sleep {sleep_seconds:.1f}s)")
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Model request failed after {MAX_RETRIES} attempts: {last_error}")


def process_abstracts(filenames):
    print(f"\n=== ABSTRACT MODE: Processing {len(filenames)} theses ===")
    completed = 0
    skipped_already_graded = 0
    skipped_no_abstract = 0
    total = len(filenames)
    already_graded = load_already_graded(OUTPUT_FILE)

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx, filename in enumerate(filenames, start=1):
        disk_graded = load_already_graded(OUTPUT_FILE)
        if disk_graded:
            already_graded.update(disk_graded)

        if filename in already_graded:
            skipped_already_graded += 1
            print(f"[{idx}/{total}] Skipping already-graded file: {filename}")
            continue

        print(f"[{idx}/{total}] Extracting abstract for {filename}")
        try:
            abstract_text, abstract_found, abstract_marker, original_chars = extract_abstract_for_filename(filename)
            if not abstract_text.strip():
                skipped_no_abstract += 1
                print(f"[{idx}/{total}] Skipping {filename}: no abstract found")
                continue

            result = grade_abstract(filename, abstract_text, original_chars, abstract_found, abstract_marker)
            if result.get("data") is not None:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed += 1
                already_graded.add(filename)
            else:
                print(f"[{idx}/{total}] Skipping {filename}: no structured score data returned")
        except Exception as exc:
            print(f"Failed grading {filename}: {exc}")

        if FORCE_GC_EACH_FILE:
            gc.collect()

        if PAUSE_BETWEEN_FILES_SECONDS > 0 and idx < total:
            time.sleep(PAUSE_BETWEEN_FILES_SECONDS)

    print(
        f"✓ Abstract grading complete! Saved {completed}/{total} results to {OUTPUT_FILE} "
        f"(skipped already-graded: {skipped_already_graded}, skipped no abstract: {skipped_no_abstract})"
    )


def display_system_prompt():
    system_prompt = ABSTRACT_GRADING_PRESET.get("llm.prediction.systemPrompt") or DEFAULT_SYSTEM_PROMPT
    response_format = build_response_format_from_preset(ABSTRACT_GRADING_PRESET)
    json_only_guard = build_json_only_output_guard(response_format)
    if json_only_guard:
        system_prompt = f"{system_prompt}\n{json_only_guard}"

    print("=" * 80)
    print("COMPLETE SYSTEM PROMPT TO BE SENT TO LLM")
    print("=" * 80)
    print()
    print(system_prompt)
    print()
    print("=" * 80)
    print(f"Total characters: {len(system_prompt)}")
    print(f"Estimated tokens: {chars_to_tokens(len(system_prompt))}")
    print("=" * 80)


def load_priority_and_canary(files):
    if PRIORITIZE_SMALLEST_FROM_CSV and PDF_INPUT_SOURCE != "gcs":
        prioritized = load_priority_filenames(PDF_CHAR_COUNT_CSV, PRIORITY_FILE_LIMIT)
        if prioritized:
            slice_start = max(0, PRIORITY_START_INDEX)
            slice_end = PRIORITY_END_INDEX if PRIORITY_END_INDEX > 0 else None
            if slice_end is not None and slice_end < slice_start:
                print(
                    f"Invalid priority slice: start_index={slice_start}, end_index={slice_end}. Resulting selection is empty."
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

    return files


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Grade thesis abstracts with streaming output.")
    parser.add_argument(
        "--system_prompt",
        action="store_true",
        help="Display the abstract-only system prompt that will be sent to the LLM.",
    )
    return parser.parse_args()


def main():
    args = parse_cli_args()
    if args.system_prompt:
        display_system_prompt()
        return 0

    files = load_input_filenames()
    files = load_priority_and_canary(files)

    if not files:
        print("No files to process.")
        return 0

    already_graded = load_already_graded(OUTPUT_FILE)
    if len(already_graded) > 0:
        print(
            f"Detected {len(already_graded)} previously graded files in {OUTPUT_FILE}. "
            "Each file will still be re-checked immediately before LLM submission."
        )

    print("Run mode: ABSTRACT-ONLY GRADING")
    process_abstracts(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
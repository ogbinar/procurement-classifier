import json
import os
import re
from typing import Any, Dict, List, Tuple

import requests

# -----------------------------------------------------------------------------
# Ollama config
# -----------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_CHAT_URL = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

# -----------------------------------------------------------------------------
# Heuristics / vocabulary
# -----------------------------------------------------------------------------
VAGUE_WORDS = {
    "item", "items", "materials", "material", "supplies", "supply",
    "equipment", "others", "misc", "miscellaneous", "stuff", "thing",
    "unit", "units", "good", "goods", "product", "products",
}

UNIT_ONLY_WORDS = {
    "mm", "cm", "m", "in", "inch", "inches", "ft", "pcs", "pc", "pack", "box",
    "g", "kg", "mg", "ml", "l", "liter", "liters", "litre", "litres",
    "oz", "lb", "amps", "amp", "v", "volts", "w", "watts", "tb", "gb",
}

FILLER_WORDS = {
    "for", "used", "use", "used for", "kind", "type", "model", "brand",
    "item", "material", "supplies", "supply",
}

TERM_NORMALIZATION = {
    "ballpen": "ballpoint pen",
    "sign pen": "marker pen",
    "bondpaper": "bond paper",
    "a4 paper": "a4 bond paper",
    "folder file": "file folder",
    "flash disk": "usb flash drive",
    "thumb drive": "usb flash drive",
    "monitor screen": "monitor",
}

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
MULTISPACE_RE = re.compile(r"\s+")
FOR_CLAUSE_RE = re.compile(r"\bfor\s+(.+)$", flags=re.IGNORECASE)
SPEC_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inch|inches|ft|pcs|pc|pack|box|g|kg|mg|ml|l|liter|liters|litre|litres|oz|lb|amps|amp|v|volts|w|watts|tb|gb)\b",
    flags=re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Basic text utilities
# -----------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_text(text: str) -> str:
    text = str(text).replace("\xa0", " ").strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def one_line_clean(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip(" \"'")


def titlecase_first(text: str) -> str:
    text = one_line_clean(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return normalize_text(str(value))


def normalize_phrase(text: str) -> str:
    return normalize_text(text).lower().strip(" ,.;:")


# -----------------------------------------------------------------------------
# Domain normalization
# -----------------------------------------------------------------------------
def normalize_use_case(text: str) -> str:
    text = normalize_phrase(text)
    text = re.sub(r"^(used for|use for|for)\s+", "", text)
    synonyms = {
        "writing office": "office writing",
        "for office writing": "office writing",
        "for office use": "office use",
        "office works": "office use",
        "office work": "office use",
    }
    return synonyms.get(text, text)


def normalize_variant(text: str) -> str:
    text = normalize_phrase(text)
    text = re.sub(r"^(variant|brand|model|type)\s*[:\-]?\s*", "", text)
    synonyms = {
        "std": "standard",
    }
    return synonyms.get(text, text)


def normalize_spec(text: str) -> str:
    text = normalize_phrase(text)
    text = re.sub(
        r"^(specification|dimensions|dimension|size|spec|grade)\s*[:\-]?\s*",
        "",
        text,
    )
    synonyms = {
        "std": "standard",
    }
    return synonyms.get(text, text)


def normalize_base_item(text: str) -> str:
    text = normalize_phrase(text)
    text = TERM_NORMALIZATION.get(text, text)
    return text


def normalize_base_item_phrase(text: str) -> str:
    text = normalize_phrase(text)
    if text in TERM_NORMALIZATION:
        return TERM_NORMALIZATION[text]
    return text


def remove_leading_filler(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(
        r"^(item|material|materials|supplies|supply|equipment|others|miscellaneous|misc)\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return normalize_text(text)


def dedupe_overlap(base_item: str, modifier: str) -> Tuple[str, str]:
    if not base_item or not modifier:
        return base_item, modifier

    base_tokens = tokenize(base_item)
    modifier_tokens = tokenize(modifier)

    if base_tokens and all(tok in modifier_tokens for tok in base_tokens):
        return "", modifier
    return base_item, modifier


# -----------------------------------------------------------------------------
# Quality assessment
# -----------------------------------------------------------------------------
def assess_description_quality(text: str) -> Dict[str, object]:
    raw = normalize_text(text)
    lowered = raw.lower()
    tokens = tokenize(raw)
    issues: List[str] = []

    if not raw:
        issues.append("description is empty")

    non_numeric_non_unit = [
        t for t in tokens
        if not t.isdigit() and t not in UNIT_ONLY_WORDS
    ]

    if len(tokens) == 1:
        issues.append("single-word description")
    elif len(tokens) < 3:
        issues.append("description has limited detail")

    if any(tok in VAGUE_WORDS for tok in tokens):
        issues.append("contains vague wording")

    if len(non_numeric_non_unit) == 0:
        issues.append("contains mostly numbers or units only")
    elif len(non_numeric_non_unit) == 1:
        issues.append("missing meaningful item detail")

    has_spec_signal = bool(SPEC_PATTERN.search(lowered)) or any(tok.isdigit() for tok in tokens)
    has_meaningful_tokens = len(non_numeric_non_unit) >= 2 and not any(tok in VAGUE_WORDS for tok in tokens)

    if has_meaningful_tokens and has_spec_signal:
        issues = [i for i in issues if i not in {"description has limited detail"}]

    if len(issues) >= 3:
        label = "LOW"
    elif len(issues) >= 1:
        label = "MEDIUM"
    else:
        label = "HIGH"

    return {"label": label, "issues": issues}


# -----------------------------------------------------------------------------
# Question selection / interaction
# -----------------------------------------------------------------------------
def choose_questions(quality_label: str, raw_text: str) -> List[Tuple[str, str]]:
    raw = normalize_text(raw_text).lower()
    tokens = tokenize(raw)

    questions: List[Tuple[str, str]] = []

    if any(tok in VAGUE_WORDS for tok in tokens):
        questions.append(("base_item", "What is the actual item name?"))

    has_spec_signal = bool(SPEC_PATTERN.search(raw)) or any(tok.isdigit() for tok in tokens)
    if not has_spec_signal:
        questions.append(("spec", "What size, grade, dimensions, or specification should be included?"))

    if "for " not in raw and quality_label == "LOW":
        questions.append(("use_case", "What will this item be used for?"))

    if quality_label in {"LOW", "MEDIUM"}:
        questions.append(("variant", "What brand, model, type, or variant is it?"))

    if not questions and quality_label == "LOW":
        questions.append(("use_case", "What will this item be used for?"))

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for field, question in questions:
        if field not in seen:
            deduped.append((field, question))
            seen.add(field)

    return deduped[:3]


def ask_questions(questions: List[Tuple[str, str]]) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    if not questions:
        return answers

    print("\nTo improve the request description, please answer a few questions.")
    print("Press Enter to skip any question.\n")

    for field, question in questions:
        answer = input(f"{question}\n> ").strip()
        answers[field] = answer

    return answers


# -----------------------------------------------------------------------------
# Rule-based parsing helpers
# -----------------------------------------------------------------------------
def extract_use_case_from_text(raw_text: str) -> Tuple[str, str]:
    raw = normalize_text(raw_text)
    match = FOR_CLAUSE_RE.search(raw)
    if not match:
        return raw, ""

    use_case = normalize_use_case(match.group(1))
    remaining = normalize_text(raw[:match.start()])
    return remaining, use_case


def extract_spec_from_text(raw_text: str) -> Tuple[str, str]:
    raw = normalize_text(raw_text)
    matches = SPEC_PATTERN.findall(raw)

    if not matches:
        return raw, ""

    specs: List[str] = []
    cleaned = raw
    for match in matches:
        specs.append(normalize_spec(match))
        cleaned = re.sub(re.escape(match), " ", cleaned, flags=re.IGNORECASE)

    cleaned = normalize_text(cleaned)
    spec = ", ".join(dict.fromkeys(specs))
    return cleaned, spec


# -----------------------------------------------------------------------------
# LLM response validation
# -----------------------------------------------------------------------------
def validate_structured_fields(structured: Dict[str, str]) -> Dict[str, str]:
    cleaned = {
        "base_item": normalize_base_item_phrase(safe_str(structured.get("base_item", ""))),
        "variant": normalize_variant(safe_str(structured.get("variant", ""))),
        "spec": normalize_spec(safe_str(structured.get("spec", ""))),
        "use_case": normalize_use_case(safe_str(structured.get("use_case", ""))),
    }

    banned_exact = {
        "item", "items", "material", "materials", "stuff", "thing",
        "others", "misc", "miscellaneous", "equipment", "supplies", "supply",
    }
    if cleaned["base_item"] in banned_exact:
        cleaned["base_item"] = ""

    cleaned["base_item"], cleaned["variant"] = dedupe_overlap(cleaned["base_item"], cleaned["variant"])
    cleaned["base_item"], cleaned["spec"] = dedupe_overlap(cleaned["base_item"], cleaned["spec"])

    if cleaned["variant"] in FILLER_WORDS:
        cleaned["variant"] = ""
    if cleaned["spec"] in FILLER_WORDS:
        cleaned["spec"] = ""

    return cleaned


def missing_critical_fields(structured: Dict[str, str]) -> List[str]:
    missing: List[str] = []
    if not structured.get("base_item", "").strip():
        missing.append("core item name")
    return missing


# -----------------------------------------------------------------------------
# Final description builder
# -----------------------------------------------------------------------------
def build_description(
    base_item: str,
    variant: str = "",
    spec: str = "",
    use_case: str = "",
) -> str:
    base_item = normalize_base_item_phrase(base_item)
    variant = normalize_variant(variant)
    spec = normalize_spec(spec)
    use_case = normalize_use_case(use_case)

    base_item, variant = dedupe_overlap(base_item, variant)
    base_item, spec = dedupe_overlap(base_item, spec)

    lead_parts = [p for p in [variant, base_item] if p]
    description = " ".join(lead_parts).strip()

    tail_parts: List[str] = []
    if spec:
        tail_parts.append(spec)
    if use_case:
        tail_parts.append(f"for {use_case}")

    if tail_parts:
        if description:
            description = f"{description}, {', '.join(tail_parts)}"
        else:
            description = ", ".join(tail_parts)

    return titlecase_first(one_line_clean(description))


# -----------------------------------------------------------------------------
# JSON extraction
# -----------------------------------------------------------------------------
def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return {}


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------
def build_structuring_prompts(raw_text: str, answers: Dict[str, str]) -> Tuple[str, str]:
    system_prompt = """
You are a procurement master-data assistant.

Your task is to convert a raw purchase request into a structured item record
for downstream procurement standardization and UNSPSC-style classification.

You must maximize consistency across semantically equivalent inputs.

NON-NEGOTIABLE RULES:
1. Do not invent facts.
2. Do not guess brand, model, size, material, or specification.
3. Only use information explicitly present in the raw text or user answers.
4. Prefer short, concrete, classifier-friendly wording.
5. Use lowercase output values only.
6. Remove filler prefixes such as: item, material, supplies, kind, type, model, brand, used for.
7. If the true core item is unclear, leave base_item empty.
8. If a field is unknown, return an empty string.
9. Put only one concept per field.
10. Return JSON only.

CANONICALIZATION RULES:
- base_item = core purchasable noun
- variant = subtype / brand / model / type only if explicitly stated
- spec = size / grade / dimensions / technical spec only if explicitly stated
- use_case = intended purpose only if explicitly stated

EXAMPLES:
Input:
raw item: pencil
variant answer: mongol
spec answer: standard
use case answer: for office writing
Output:
{"base_item":"pencil","variant":"mongol","spec":"standard","use_case":"office writing"}

Input:
raw item: brand: mongol pencil specification: standard used for office writing
Output:
{"base_item":"pencil","variant":"mongol","spec":"standard","use_case":"office writing"}

Input:
raw item: office supplies
Output:
{"base_item":"","variant":"","spec":"","use_case":""}

Return JSON with exactly these keys:
{
  "base_item": "",
  "variant": "",
  "spec": "",
  "use_case": ""
}
    """.strip()

    user_prompt = f"""
RAW INPUT
raw_item_description: {raw_text}

USER ANSWERS
base_item_answer: {answers.get('base_item', '')}
variant_answer: {answers.get('variant', '')}
spec_answer: {answers.get('spec', '')}
use_case_answer: {answers.get('use_case', '')}

Return the JSON now.
    """.strip()

    return system_prompt, user_prompt


# -----------------------------------------------------------------------------
# Ollama integration
# -----------------------------------------------------------------------------
def call_ollama_structurer(raw_text: str, answers: Dict[str, str]) -> Dict[str, str]:
    system_prompt, user_prompt = build_structuring_prompts(raw_text, answers)

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0,
            "top_p": 1,
        },
    }

    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        content = ""
        if isinstance(data, dict):
            content = safe_str(data.get("message", {}).get("content", ""))

        obj = extract_json_object(content)

        structured = {
            "base_item": safe_str(obj.get("base_item", "")),
            "variant": safe_str(obj.get("variant", "")),
            "spec": safe_str(obj.get("spec", "")),
            "use_case": safe_str(obj.get("use_case", "")),
        }

        structured = validate_structured_fields(structured)

        # conservative repair only after LLM call
        repaired = fallback_structure(raw_text, answers)

        if not structured["base_item"]:
            structured["base_item"] = repaired.get("base_item", "")
        if not structured["variant"]:
            structured["variant"] = repaired.get("variant", "")
        if not structured["spec"]:
            structured["spec"] = repaired.get("spec", "")
        if not structured["use_case"]:
            structured["use_case"] = repaired.get("use_case", "")

        structured = validate_structured_fields(structured)

        if not any(structured.values()):
            raise ValueError("Empty structured result from Ollama")

        return structured

    except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
        print(f"\n[WARN] Ollama request failed: {e}")
        print("[WARN] Falling back to simple rule-based structuring.")
        return fallback_structure(raw_text, answers)


# -----------------------------------------------------------------------------
# Fallback rule-based structure
# -----------------------------------------------------------------------------
def fallback_structure(raw_text: str, answers: Dict[str, str]) -> Dict[str, str]:
    raw = normalize_text(raw_text)
    raw = remove_leading_filler(raw)

    base_item = normalize_base_item_phrase(answers.get("base_item", ""))
    variant = normalize_variant(answers.get("variant", ""))
    spec = normalize_spec(answers.get("spec", ""))
    use_case = normalize_use_case(answers.get("use_case", ""))

    working_text = raw

    if not use_case:
        working_text, extracted_use_case = extract_use_case_from_text(working_text)
        use_case = extracted_use_case

    if not spec:
        working_text, extracted_spec = extract_spec_from_text(working_text)
        spec = extracted_spec

    working_text = normalize_text(working_text).lower().strip(" ,;-:")
    working_text = remove_leading_filler(working_text)
    working_text = normalize_base_item_phrase(working_text)

    if not base_item:
        base_item = working_text

    structured = {
        "base_item": base_item,
        "variant": variant,
        "spec": spec,
        "use_case": use_case,
    }
    return validate_structured_fields(structured)


# -----------------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------------
def print_quality(result: Dict[str, object]) -> None:
    print(f"\nDescription quality: {result['label']}")
    issues = result.get("issues", [])
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")


def print_structured(structured: Dict[str, str]) -> None:
    print("\nStructured fields:\n")
    print(json.dumps(structured, indent=2, ensure_ascii=False))


def print_missing_fields(structured: Dict[str, str]) -> None:
    missing = missing_critical_fields(structured)
    if not missing:
        return

    print("\nStill missing critical information:")
    for field in missing:
        print(f"- {field}")
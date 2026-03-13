# test_semantic_consistency.py

import json
import unittest
from typing import Dict, List

from utils import (
    build_description,
    call_ollama_structurer,
    normalize_base_item_phrase,
    normalize_spec,
    normalize_use_case,
    normalize_variant,
)


def canonicalize_structured(structured: Dict[str, str]) -> Dict[str, str]:
    """Normalize fields so minor formatting differences don't fail the test."""
    return {
        "base_item": normalize_base_item_phrase(structured.get("base_item", "")),
        "variant": normalize_variant(structured.get("variant", "")),
        "spec": normalize_spec(structured.get("spec", "")),
        "use_case": normalize_use_case(structured.get("use_case", "")),
    }


def run_pipeline(raw_text: str, answers: Dict[str, str] | None = None) -> Dict[str, str]:
    answers = answers or {}
    structured = call_ollama_structurer(raw_text, answers)
    structured = canonicalize_structured(structured)

    description = build_description(
        base_item=structured.get("base_item", ""),
        variant=structured.get("variant", ""),
        spec=structured.get("spec", ""),
        use_case=structured.get("use_case", ""),
    )

    return {
        **structured,
        "description": description,
    }


class TestSemanticConsistency(unittest.TestCase):
    def test_pencil_office_use_consistency(self):
        # Same meaning, slightly different phrasing
        examples: List[Dict[str, Dict[str, str] | str]] = [
            {
                "raw_text": "pencil for office use",
                "answers": {},
            },
            {
                "raw_text": "office pencil",
                "answers": {"use_case": "office use"},
            },
            {
                "raw_text": "item: pencil used for office work",
                "answers": {},
            },
            {
                "raw_text": "pencil",
                "answers": {"use_case": "office use"},
            },
            {
                "raw_text": "material: pencil for writing in the office",
                "answers": {},
            },
        ]

        results = []
        for ex in examples:
            result = run_pipeline(ex["raw_text"], ex["answers"])
            results.append(result)

        print("\n=== Semantic Consistency Results ===")
        for i, result in enumerate(results, start=1):
            print(f"\nCase {i}")
            print(json.dumps(result, indent=2, ensure_ascii=False))

        # Choose the first output as the reference
        reference = results[0]

        # Compare only the fields you care most about being consistent
        for i, result in enumerate(results[1:], start=2):
            self.assertEqual(
                result["base_item"],
                reference["base_item"],
                msg=f"base_item mismatch in case {i}",
            )
            self.assertEqual(
                result["use_case"],
                reference["use_case"],
                msg=f"use_case mismatch in case {i}",
            )
            self.assertEqual(
                result["description"],
                reference["description"],
                msg=f"description mismatch in case {i}",
            )


if __name__ == "__main__":
    unittest.main()
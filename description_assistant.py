#!/usr/bin/env python3
"""
description_assistant.py

Minimal CLI MVP:
- checks if a procurement description is weak/vague
- asks up to 3 simple clarifying questions
- uses Ollama to structure and normalize into procurement-friendly fields
- builds a consistent description for downstream UNSPSC classification

Run:
    python description_assistant.py

Optional environment variables:
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_MODEL=qwen3:1.7b
"""

import sys

from utils import (
    assess_description_quality,
    ask_questions,
    build_description,
    call_ollama_structurer,
    choose_questions,
    print_missing_fields,
    print_quality,
    print_structured,
)


def main() -> None:
    print("=== Procurement Description Assistant ===")

    while True:
        raw_text = input("\nEnter item description (blank to exit):\n> ").strip()

        if not raw_text:
            print("Goodbye.")
            break

        quality = assess_description_quality(raw_text)
        print_quality(quality)

        questions = choose_questions(quality["label"], raw_text)
        answers = ask_questions(questions)

        structured = call_ollama_structurer(raw_text, answers)

        improved = build_description(
            base_item=structured.get("base_item", raw_text),
            variant=structured.get("variant", ""),
            spec=structured.get("spec", ""),
            use_case=structured.get("use_case", ""),
        )

        print_structured(structured)
        print_missing_fields(structured)

        print("\nSuggested description:\n")
        print(improved if improved else raw_text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
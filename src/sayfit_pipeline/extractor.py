from __future__ import annotations

import json
import os
import re
from typing import Any

from .models import ExtractedItem

SPLIT_RE = re.compile(r",|\band\b|\bwith\b", flags=re.IGNORECASE)
AMOUNT_UNIT_NAME_RE = re.compile(
    r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>kg|g|mg|l|ml|cup|cups|tbsp|tsp|slice|slices|serving|servings|piece|pieces|pcs|egg|eggs|banana|bananas)?\s*(?P<name>[a-zA-Z][a-zA-Z\-\s']+)",
    flags=re.IGNORECASE,
)
COUNT_NAME_RE = re.compile(r"(?P<amount>\d+(?:\.\d+)?)\s+(?P<name>[a-zA-Z][a-zA-Z\-\s']+)", flags=re.IGNORECASE)
WORD_UNIT_NAME_RE = re.compile(
    r"(?P<amount_word>half|quarter|one|two|three|four|five|a|an)\s+"
    r"(?P<unit>handful|hand full|pinch|dash|slice|slices|serving|servings|piece|pieces|egg|eggs)\s*(?:of)?\s+"
    r"(?P<name>[a-zA-Z][a-zA-Z\-\s']+)",
    flags=re.IGNORECASE,
)
WORD_NAME_RE = re.compile(
    r"(?P<amount_word>half|quarter|one|two|three|four|five)\s+(?:of\s+)?(?P<name>[a-zA-Z][a-zA-Z\-\s']+)",
    flags=re.IGNORECASE,
)

STOP_PREFIXES = {
    "i ate",
    "i had",
    "for breakfast",
    "for lunch",
    "for dinner",
    "lunch was",
    "breakfast was",
    "dinner was",
    "snack",
}
STOP_SUFFIXES = {
    "for breakfast",
    "for lunch",
    "for dinner",
    "at breakfast",
    "at lunch",
    "at dinner",
}
LEADING_DESCRIPTORS = {
    "creamy",
    "green",
    "red",
    "roasted",
    "grilled",
    "steamed",
    "cooked",
    "fresh",
    "raw",
}
AMOUNT_WORDS = {
    "a": 1.0,
    "an": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "half": 0.5,
    "quarter": 0.25,
}
QUERY_NOISE_TOKENS = {
    "of",
    "handful",
    "hand",
    "full",
    "pinch",
    "dash",
    "serving",
    "servings",
    "slice",
    "slices",
    "piece",
    "pieces",
}


class ItemExtractor:
    def __init__(self) -> None:
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        llm_flag = os.getenv("SAYFIT_ENABLE_LLM")
        if llm_flag is None:
            self.enable_llm = bool(self.groq_api_key)
        else:
            self.enable_llm = llm_flag == "1"
        self.groq_model = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")

    def extract(self, text: str) -> list[ExtractedItem]:
        if self.enable_llm and self.groq_api_key:
            parsed = self._extract_with_groq(text)
            if parsed:
                return parsed
        return self._extract_rule_based(text)

    def _extract_with_groq(self, text: str) -> list[ExtractedItem]:
        try:
            from openai import OpenAI
        except Exception:
            return []

        try:
            client = OpenAI(api_key=self.groq_api_key, base_url="https://api.groq.com/openai/v1")
            prompt = (
                "Extract food items from the user sentence. Return JSON only with schema: "
                '{"items":[{"name":str,"amount":number|null,"unit":str|null}]}. '
                "Do not invent missing amounts or units."
            )
            resp = client.chat.completions.create(
                model=self.groq_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            items = []
            for raw in data.get("items", []):
                name = str(raw.get("name", "")).strip()
                if not name:
                    continue
                amount = raw.get("amount")
                unit = raw.get("unit")
                items.append(
                    ExtractedItem(
                        name=name,
                        amount=float(amount) if amount is not None else None,
                        unit=(str(unit).lower() if unit else None),
                    )
                )
            return items
        except Exception:
            return []

    def _clean_phrase(self, phrase: str) -> str:
        cleaned = phrase.strip(" .:;!?")
        lowered = cleaned.lower().strip()
        for prefix in STOP_PREFIXES:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip(" .:;!?")
                lowered = cleaned.lower().strip()
        for suffix in STOP_SUFFIXES:
            if lowered.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip(" .:;!?")
                lowered = cleaned.lower().strip()
        return cleaned

    def _extract_rule_based(self, text: str) -> list[ExtractedItem]:
        chunks = [self._clean_phrase(p) for p in SPLIT_RE.split(text) if p.strip()]
        items: list[ExtractedItem] = []

        for chunk in chunks:
            if not chunk:
                continue

            m = AMOUNT_UNIT_NAME_RE.search(chunk)
            if m:
                amount = float(m.group("amount"))
                unit = (m.group("unit") or "").lower().strip() or None
                name = m.group("name").strip()
                items.append(ExtractedItem(name=name, amount=amount, unit=unit))
                continue

            w = WORD_UNIT_NAME_RE.search(chunk)
            if w:
                amount_word = w.group("amount_word").lower().strip()
                amount = AMOUNT_WORDS.get(amount_word)
                unit = (w.group("unit") or "").lower().strip() or None
                name = w.group("name").strip()
                items.append(ExtractedItem(name=name, amount=amount, unit=unit))
                continue

            wn = WORD_NAME_RE.search(chunk)
            if wn:
                amount_word = wn.group("amount_word").lower().strip()
                amount = AMOUNT_WORDS.get(amount_word)
                name = wn.group("name").strip()
                items.append(ExtractedItem(name=name, amount=amount, unit="piece"))
                continue

            c = COUNT_NAME_RE.search(chunk)
            if c:
                amount = float(c.group("amount"))
                name = c.group("name").strip()
                items.append(ExtractedItem(name=name, amount=amount, unit="piece"))
                continue

            items.append(ExtractedItem(name=chunk, amount=None, unit=None))

        # Light normalization for singular/plural
        normalized: list[ExtractedItem] = []
        for item in items:
            name = re.sub(r"\s+", " ", item.name.lower()).strip()
            tokens = name.split()
            while len(tokens) > 1 and tokens[0] in LEADING_DESCRIPTORS:
                tokens = tokens[1:]
            while len(tokens) > 1 and tokens[0] in QUERY_NOISE_TOKENS:
                tokens = tokens[1:]
            if len(tokens) > 2:
                tokens = [t for t in tokens if t not in QUERY_NOISE_TOKENS]
                if not tokens:
                    tokens = name.split()
            name = " ".join(tokens)
            if name.endswith("s") and len(name) > 3 and not name.endswith("ss"):
                # Keep both plural and singular food words reasonable for search.
                name = name[:-1]
            normalized.append(ExtractedItem(name=name, amount=item.amount, unit=item.unit))
        return normalized

"""
Step 1 – Item Extraction via OpenAI LLM.

Takes raw text (e.g. "I ate a pepperoni pizza and 3 eggs")
and returns structured FoodItem list using a strict JSON schema.
"""

from __future__ import annotations

import json
from datetime import datetime

from openai import OpenAI

from sayfit.models import FoodItem, ParsedMeal

# System prompt that enforces structured output
EXTRACTION_SYSTEM_PROMPT = """\
You are a food-intake parser. The user will describe what they ate in natural language.

Your job: extract EVERY food item mentioned and return a JSON object matching this schema exactly:

{
  "items": [
    {
      "name": "<food name, lowercase>",
      "amount": <number or null if not mentioned>,
      "unit": "<g|kg|ml|l|piece|slice|serving|cup|tbsp|tsp|null>"
    }
  ]
}

Rules:
- Always use lowercase food names.
- Keep the food name in its NATURAL plural/singular form as the user said it.
  "50 grapes" → name="grapes", amount=50, unit="piece".
  "3 eggs" → name="eggs", amount=3, unit="piece".
  "a banana" → name="banana", amount=1, unit="piece".
- If the user says "oatmeal" with no quantity, amount=null, unit=null.
- If the user says "2 slices of bread", name="bread", amount=2, unit="slice".
- "200ml almond milk" → name="almond milk", amount=200, unit="ml".
- "a glass of red wine" → name="red wine", amount=1, unit="serving".
- Prefer specific names: "pepperoni pizza" not just "pizza".
- Do NOT invent nutritional values. Only extract items.
- Return ONLY valid JSON, no extra text.
"""


def extract_items(
    text: str,
    api_key: str,
    model: str = "gpt-5.2",
    timestamp: str | None = None,
) -> ParsedMeal:
    """Call OpenAI Responses API to parse free-text into structured food items."""
    client = OpenAI(api_key=api_key)

    prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nUser input: {text}\n\nReturn ONLY valid JSON."

    resp = client.responses.create(
        model=model,
        input=prompt,
    )

    raw = resp.output_text.strip()
    # Strip markdown fences if the model wraps output in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    data = json.loads(raw)

    items = [FoodItem(**item) for item in data.get("items", [])]

    ts = timestamp or datetime.now().isoformat(timespec="seconds")
    return ParsedMeal(items=items, timestamp=ts)

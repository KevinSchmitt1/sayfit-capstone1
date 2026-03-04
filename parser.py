import json
import re
from typing import List

def extract_quantities(text: str):
    """
    Extrahiert Mengenangaben wie 100g, 200ml, 1kg, 1l
    """
    pattern = r'(\d+(?:\.\d+)?)\s?(g|kg|ml|l)'
    return re.findall(pattern, text.lower())


def extract_food_entities(text: str, food_db: dict) -> List[str]:
    """
    Extrahiert Lebensmittel aus dem Text basierend auf der Food-Datenbank.
    Liefert die Lebensmittel in der Reihenfolge, wie sie im Text vorkommen.
    """
    foods_found = []
    text_lower = text.lower()
    for food in food_db.keys():
        if food in text_lower:
            foods_found.append(food)
    return foods_found


def parse_food_from_text(text: str, food_db: dict):
    """
    Parst Text zu Items mit name, amount, unit.
    Die Reihenfolge der Mengen und Lebensmittel wird beibehalten.
    """
    quantities = extract_quantities(text)
    foods = extract_food_entities(text, food_db)

    items = []

    # Wir matchen Mengen mit Lebensmitteln in der Reihenfolge des Textes
    for i, quantity in enumerate(quantities):
        amount = float(quantity[0])
        unit = quantity[1]

        # Name vom Food-Entity nehmen, falls vorhanden, sonst 'unknown'
        name = foods[i] if i < len(foods) else "unknown"

        items.append({
            "name": name,
            "amount": amount,
            "unit": unit
        })

    return {"items": items}


def parse_json_file(input_path: str, output_path: str, db_path: str):
    """
    Liest Input JSON ein, parst Text und schreibt parsed JSON
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    text = data["text"]

    # Food-Datenbank laden
    with open(db_path, "r") as f:
        food_db = json.load(f)

    parsed = parse_food_from_text(text, food_db)

    # Timestamp aus Input übernehmen, falls vorhanden
    if "timestamp" in data:
        parsed["timestamp"] = data["timestamp"]

    with open(output_path, "w") as f:
        json.dump(parsed, f, indent=2)
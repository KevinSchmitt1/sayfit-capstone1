from parser import parse_json_file
from nutrition_calculator import enrich
from food_matcher import FoodMatcher
import os
import json
import glob

def run_pipeline(input_file):
    os.makedirs("outputs", exist_ok=True)

    # Input JSON laden, um Timestamp zu übernehmen
    with open(input_file, "r") as f:
        input_data = json.load(f)

    parsed_file = "outputs/parsed_" + os.path.basename(input_file)
    final_file = "outputs/final_" + os.path.basename(input_file)
    db_path = "data/food_database.json"

    # 1. Text parsen
    parse_json_file(input_file, parsed_file, db_path)

    # 2. Matcher initialisieren
    matcher = FoodMatcher(db_path)

    # 3. Nährwerte berechnen
    enrich(parsed_file, db_path, final_file, matcher)

    # 4. Timestamp ins Output schreiben
    with open(final_file, "r") as f:
        output_data = json.load(f)

    output_data["timestamp"] = input_data.get("timestamp", None)

    with open(final_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Pipeline finished. Output written to {final_file}")


if __name__ == "__main__":
    # alle Testdateien automatisch durchlaufen
    for input_file in sorted(glob.glob("input_samples/*.json")):
        run_pipeline(input_file)
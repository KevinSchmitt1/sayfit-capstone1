import json

def enrich(parsed_path, db_path, output_path, matcher):
    with open(parsed_path, "r") as f:
        data = json.load(f)

    with open(db_path, "r") as f:
        db = json.load(f)

    totals = {"calories": 0, "protein": 0, "fiber": 0}

    for item in data["items"]:
        # Food semantisch matchen
        matched_name = matcher.match(item["name"])
        item["matched_name"] = matched_name

        nutrition_per_100g = db[matched_name]
        factor = item["amount"] / 100

        nutrition = {
            "calories": round(nutrition_per_100g["calories"] * factor, 2),
            "protein": round(nutrition_per_100g["protein"] * factor, 2),
            "fiber": round(nutrition_per_100g["fiber"] * factor, 2)
        }

        item["nutrition"] = nutrition

        totals["calories"] += nutrition["calories"]
        totals["protein"] += nutrition["protein"]
        totals["fiber"] += nutrition["fiber"]

    data["totals"] = totals

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
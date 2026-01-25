import json
from pathlib import Path


def check_json_files(directory):
    path = Path(directory)
    for json_file in path.glob("**/*.json"):  # Search recursively
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {json_file}")
            print(f"   Error: {e}")
        except Exception as e:
            print(f"❌ System Error reading {json_file}: {e}")


check_json_files("./result/raw/drone")

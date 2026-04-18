from pathlib import Path
import json
from collections import defaultdict

BASE_DIR = Path(__file__).parents[0].parents[0].resolve()
DATA_DIR = BASE_DIR / "sample_data"
CATEGORY_PATH = DATA_DIR / "categories.json"
MAP_PATH = DATA_DIR / "map.json"

class LabelConverter():
    def __init__(self, category_path, map_path):
        self.category_path = category_path
        self.map_path = map_path

    def _get_category_map(self):
        category_map = {}
        
        with open(self.category_path, 'r', encoding="utf-8") as inf:
            categories = json.load(inf)
            sorted_categories = sorted(list(map(lambda x: x.lower(), categories)))

            for index, item in enumerate(sorted_categories):
                category_map[item] = index

        return category_map

    def _write_category_map(self, category_map):
        with open(self.map_path, "w", encoding="utf-8") as outf:
            data = json.dumps(category_map)

            outf.write(data)


if __name__ == "__main__":
    converter = LabelConverter(CATEGORY_PATH, MAP_PATH)
    category_map = converter._get_category_map()
    converter._write_category_map(category_map)
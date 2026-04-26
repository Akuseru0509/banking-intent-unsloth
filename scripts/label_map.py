from pathlib import Path
import json

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

    def _get_map(self):
        try:
            with open(self.map_path, "r", encoding="utf-8") as inf:
                category_map = json.load(inf)

            return dict(category_map)
        
        except FileNotFoundError:
            raise ValueError(f"No file found at {self.map_path}")
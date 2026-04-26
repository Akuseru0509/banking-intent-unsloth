from datasets import Dataset, load_dataset, ClassLabel

class DataProcessor():
    def __init__(self, category_map):
        self.category_map = category_map

    def _load(self, data_path) -> Dataset:
        try:
            if not self.category_map:
                raise ValueError(f"Error getting category_map")
            
            dataset = load_dataset("csv", data_files = data_path)["train"]

            def create_label(example):
                key = example["category"].strip().lower()
                example["label"] = self.category_map[key]

                return example
            
            dataset = dataset.map(create_label)

            return dataset

        except FileNotFoundError:
            raise ValueError(f"Error finding files at {self.data_path}")
        
    @staticmethod
    def _split(train_set: Dataset) -> tuple[Dataset, Dataset]:
        try:
            num_classes = len(set(train_set["label"]))

            train_set = train_set.cast_column(
                "label",
                ClassLabel(num_classes=num_classes)
            )

            split = train_set.train_test_split(
                test_size = 0.1,
                seed = 42,
                stratify_by_column = "label"
            )

            validation, train = split["test"], split["train"]

            return train, validation
        except Exception as e:
            raise ValueError(f"Error: {e}")
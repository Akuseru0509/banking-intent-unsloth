from pathlib import Path
import yaml
from unsloth import FastLanguageModel

from label_map import LabelConverter

BASE_DIR = Path(__file__).parents[0].parents[0]
DATA_DIR = BASE_DIR / "sample_data"
CONFIG_DIR = BASE_DIR / "configs"
MODEL_PATH = BASE_DIR / "model"

class IntentClassification:
    def __init__(self, model_path, yaml_path):
        self.model_path = model_path
        
        with open(yaml_path, "r", encoding="utf-8") as inf:
            self.configurations = yaml.safe_load(inf)

        label_converter = LabelConverter(
            DATA_DIR / "categories.json",
            DATA_DIR / "map.json"
        )

        category_map = label_converter._get_map()
        self.reverse_map = {v: k for k, v in category_map.items()}

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_path,
            max_seq_length = self.configurations["max_seq_length"],
            dtype = self.configurations["dtype"],
            load_in_4bit = self.configurations["load_in_4bit"]
        )

        FastLanguageModel.for_inference(self.model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prompt = """
        ### Instruction:
        Classify the intent of the following banking request.

        ### Input:
        {}

        ### Response:
        Answer:"""

    def __call__(self, message):
        prompt = self.prompt.format(message)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        input_length = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.configurations["max_new_tokens"],
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        prediction_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()

        import re
        match = re.search(r"\d+", prediction_text)

        if match:
            pred_label = int(match.group())
        else:
            return "unknown"

        return self.reverse_map.get(pred_label, "unknown")
    
if __name__ == "__main__":
    classifier = IntentClassification(MODEL_PATH, CONFIG_DIR / "inference.yaml")
    message = "Am I able to get a card in EU?"
    predicted_label = classifier.__call__(message)

    print(predicted_label)
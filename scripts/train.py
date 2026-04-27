from pathlib import Path
import yaml
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from preprocess import DataProcessor
from label_map import LabelConverter

class IntentTrainer():
    def __init__(self):
        self.configurations = {}
        self.prompt = """
            ### Instruction:
            Classify the intent of the following banking request.

            ### Input:
            {}

            ### Response:
            Answer: <|label|> {}
        """

    def _load_config(self, yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as inf:
                self.configurations = yaml.safe_load(inf)

        except FileNotFoundError:
            raise ValueError(f"No file found at {yaml_path}")
        
    def _init_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.configurations["model_name"],
            max_seq_length = self.configurations["max_seq_length"],
            load_in_4bit = self.configurations["load_in_4bit"],
            dtype = self.configurations["dtype"]
        )

        model = FastLanguageModel.get_peft_model(
            model = model,
            r = self.configurations["lora_r"],
            target_modules = self.configurations["target_modules"],
            lora_alpha = self.configurations["lora_alpha"],
            lora_dropout = self.configurations["lora_dropout"],
            bias = self.configurations["bias"],
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = self.configurations["use_rslora"],
            loftq_config = self.configurations["loftq_config"]
        )

        return model, tokenizer
    
    def _format_inputs(self, tokenizer, train_set, validation_set, test_set):
        try:
            def format_prompts(examples):
                texts = []
                for input_text, output_text in zip(examples["texts"], examples["labels"]):
                    text = self.prompt.format(
                        input_text,
                        str(output_text)
                    ) + tokenizer.eos_token

                    texts.append(text)

                return {"texts": texts}
            
            train_set = train_set.map(format_prompts, batched=True)
            validation_set = validation_set.map(format_prompts, batched=True)
            test_set = test_set.map(format_prompts, batched=True)

        except Exception as e:
            raise ValueError(f"Error: {e}")

    def _tokenize(self, tokenizer, train_set, validation_set):
        try:
            def tokenize(example):
                text = example["texts"]

                marker = "<|label|>"
                start_char = text.find(marker)
                if start_char == -1:
                    raise ValueError(f"Marker not found in: {text}")

                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=256,
                    return_offsets_mapping=True,
                )

                labels = tokenized["input_ids"].copy()
                offsets = tokenized["offset_mapping"]

                start_token = None
                for i, (s, e) in enumerate(offsets):
                    if s <= start_char < e:
                        start_token = i + 1 
                        break

                if start_token is None:
                    raise ValueError("Token alignment failed")

                for i in range(start_token):
                    labels[i] = -100

                tokenized["labels"] = labels
                tokenized.pop("offset_mapping")

                return tokenized
            
            train_set = train_set.map(tokenize)
            validation_set = validation_set.map(tokenize)

        except Exception as e:
            raise ValueError(f"Error: {e}")
        
    def _get_training_args(self):
        return SFTConfig(
            dataset_num_proc = 2,
            packing = False,

            per_device_train_batch_size = self.configurations["batch_size"],
            gradient_accumulation_steps = self.configurations["gradient_accumulation"],

            warmup_steps = self.configurations["logging_steps"],
            eval_strategy = "steps",
            eval_steps = self.configurations["eval_steps"],
            save_strategy = "steps",
            save_steps = self.configurations["save_steps"],
            load_best_model_at_end = True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            optim = self.configurations["optimizer"],
            weight_decay = self.configurations["decay"],
            lr_scheduler_type = self.configurations["scheduler"],

            seed = self.configurations["seed"],
            output_dir = self.configurations["output_dir"] 
        )
    
    def _get_trainer(self, model, tokenizer, train_set, validation_set, training_args):
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_set,
            eval_dataset=validation_set,
            args=training_args
        )

    def _evaluate(self, model, tokenizer, test_set):
        from tqdm.notebook import trange

        y_true = []
        y_pred = []

        total_samples = len(test_set)

        for i in trange(total=total_samples, desc="Evaluating"):
            example = test_set[i]
            text = example["text"]
            label = int(example["label"])

            prompt = """### Instruction:
            Classify the intent of the following banking request.
            
            ### Input:
            {}
            
            ### Response:
            Answer: <|label|>"""

            promtp = prompt.format(
                text
            )

            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

            input_length = inputs.input_ids.shape[1]

            output = model.generate(
                **inputs,
                max_new_tokens=self.configurations["max_new_tokens"],
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

            predicted_text = tokenizer.decode(
                output[0][input_length:],
                skip_special_tokens=True
            ).strip()

            numeric_id = ''.join(filter(str.isdigit, predicted_text))

            if numeric_id == "":
                pred_label = -1
            else:
                pred_label = int(numeric_id)

            y_true.append(label)
            y_pred.append(pred_label)
        
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        print(f"\nFinal Test Set Results ({total_samples} samples):")
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Macro F1:  {macro_f1 * 100:.2f}%")

        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    
    def _pipeline(self):
        try:
            BASE_DIR = Path(__file__).parents[0].parents[0]
            DATA_DIR = BASE_DIR / "sample_data"
            CONFIG_DIR = BASE_DIR / "configs"

            label_converter = LabelConverter(DATA_DIR / "categories.json", DATA_DIR / "map.json")
            category_map = label_converter._get_map()

            data_processor = DataProcessor(category_map=category_map)
            train_set = data_processor._load(DATA_DIR / "train.csv")
            test_set = data_processor._load(DATA_DIR / "test.csv")

            train_set, validation_set = data_processor._split(train_set=train_set)
            self._load_config(CONFIG_DIR / "train.yaml")

            model, tokenizer = self._init_model()
            FastLanguageModel.for_training(model)
            
            self._format_inputs(
                tokenizer=tokenizer, 
                train_set=train_set, 
                validation_set=validation_set,
                test_set=test_set
            )

            self._tokenize(
                tokenizer=tokenizer,
                train_set=train_set,
                validation_set=validation_set
            )

            training_args = self._get_training_args()
            
            trainer = self._get_trainer(
                model=model,
                tokenizer=tokenizer,
                validation_set=validation_set,
                training_args=training_args
            )
            
            trainer.train()

            self._evaluate(model, tokenizer, test_set)

        except Exception as e:
            raise ValueError(f"Error: {e}")
        

if __name__ == "__main__":
    intent_trainer = IntentTrainer()
    intent_trainer._pipeline()
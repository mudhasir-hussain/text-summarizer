import os
from textsummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from textsummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        model_inputs = self.tokenizer(
            example_batch['dialogue'],
            max_length=1024,
            truncation=True
        )

        labels = self.tokenizer(
            text_target=example_batch['summary'],
            max_length=128,
            truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        logger.info(f"Loaded dataset from {self.config.data_path}")
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        
        save_path = os.path.join(self.config.root_dir, "samsum_dataset")
        dataset_samsum_pt.save_to_disk(save_path)
        logger.info(f"Transformed dataset saved to {save_path}")

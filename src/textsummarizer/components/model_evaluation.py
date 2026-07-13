import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate
from textsummarizer.entity import ModelEvaluationConfig
from textsummarizer.logging import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self, dataset, metric, model, tokenizer, batch_size=2, device="cpu",
        column_text="dialogue", column_summary="summary"
    ):
        model.eval()
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            beams = 1 if device == "cpu" else 8
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=beams,
                max_length=128
            )

            decoded_summaries = tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            references = [[ref] for ref in target_batch]

            metric.add_batch(
                predictions=decoded_summaries,
                references=references
            )

        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for model evaluation.")

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        # Load dataset
        dataset_samsum = load_from_disk(self.config.data_path)

        test_dataset = dataset_samsum['test']
        if device == "cpu":
            logger.info("Evaluating on a small subset (first 2 samples) of the test set since we are on CPU.")
            test_dataset = test_dataset.select(range(2))

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = evaluate.load('rouge')

        score = self.calculate_metric_on_test_ds(
            test_dataset,
            rouge_metric,
            model_pegasus,
            tokenizer,
            batch_size=2,
            device=device,
            column_text='dialogue',
            column_summary='summary'
        )

        # Convert scores to dictionary of float values
        rouge_dict = {}
        for rn in rouge_names:
            val = score.get(rn, 0.0)
            if hasattr(val, 'mid') and hasattr(val.mid, 'fmeasure'):
                rouge_dict[rn] = float(val.mid.fmeasure)
            elif isinstance(val, dict) and 'fmeasure' in val:
                rouge_dict[rn] = float(val['fmeasure'])
            else:
                rouge_dict[rn] = float(val)

        # Save metrics to JSON file
        with open(self.config.metric_file_name, "w") as f:
            json.dump(rouge_dict, f, indent=4)

        logger.info(f"Evaluation metrics saved to {self.config.metric_file_name}")

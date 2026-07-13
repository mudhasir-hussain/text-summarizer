import os
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textsummarizer.entity import ModelTrainerConfig
from textsummarizer.logging import logger
import torch

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for training")

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Set max_steps if training on CPU and max_steps is -1 to prevent long CPU training sessions.
        max_steps = self.config.max_steps
        if device in ["cpu", "mps"] and max_steps == -1:
            logger.warning("Training on CPU with max_steps = -1 will take a very long time! Setting max_steps = 2 for quick verification.")
            max_steps = 2

        use_cpu = True if device == "cpu" else False
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_steps=max_steps,
            use_cpu=use_cpu
        )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        # Save model and tokenizer
        model_path = os.path.join(self.config.root_dir, "pegasus-samsum-model")
        tokenizer_path = os.path.join(self.config.root_dir, "tokenizer")
        
        model_pegasus.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Tokenizer saved to {tokenizer_path}")

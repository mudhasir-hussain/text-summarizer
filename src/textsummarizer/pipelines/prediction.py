import os
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from textsummarizer.config.configuration import ConfigurationManager
from textsummarizer.logging import logger
import torch

class PredictionPipeline:
    # Class-level variables to cache the preloaded model and tokenizer
    _model = None
    _tokenizer = None
    _loaded_path = None

    def __init__(self):
        try:
            self.config = ConfigurationManager().get_model_evaluation_config()
            self.use_config = True
        except Exception as e:
            logger.warning(f"Could not load configuration manager: {e}. Using default paths.")
            self.use_config = False

    def predict(self, text):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
            
        # Default model and tokenizer ID/path
        model_path = os.environ.get("PEGASUS_MODEL_ID", "transformersbook/pegasus-samsum")
        tokenizer_path = "transformersbook/pegasus-samsum" # Always load standard tokenizer

        if self.use_config and not os.environ.get("PEGASUS_MODEL_ID"):
            config_model_path = str(self.config.model_path)
            config_tokenizer_path = str(self.config.tokenizer_path)
            if os.path.exists(config_model_path) and os.path.exists(config_tokenizer_path):
                model_path = config_model_path
                tokenizer_path = config_tokenizer_path

        # Check if cache is empty or path changed, and load model
        if PredictionPipeline._model is None or PredictionPipeline._loaded_path != model_path:
            logger.info(f"Preloading fine-tuned model and tokenizer from: {model_path} into memory cache...")
            PredictionPipeline._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            PredictionPipeline._model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            PredictionPipeline._loaded_path = model_path
        else:
            logger.info("Reusing preloaded fine-tuned model and tokenizer from memory cache.")

        tokenizer = PredictionPipeline._tokenizer
        model = PredictionPipeline._model

        logger.info(f"Dialogue to summarize: \n{text}")
        
        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                length_penalty=1.0,
                num_beams=2,
                max_length=256,
                no_repeat_ngram_size=3
            )
            
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Clean up Pegasus tokenizer newlines
        output = output.replace("<n>", "\n")
        
        logger.info(f"Model response: \n{output}")
        return output

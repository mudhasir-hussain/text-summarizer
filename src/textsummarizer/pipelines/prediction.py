import os
from huggingface_hub import InferenceClient
from textsummarizer.logging import logger

class PredictionPipeline:
    def __init__(self):
        self.use_config = False

    def predict(self, text):
        model_id = os.environ.get("PEGASUS_MODEL_ID", "philschmid/bart-large-cnn-samsum")
        hf_token = os.environ.get("HF_TOKEN")
        
        logger.info(f"Predicting summary via HF InferenceClient for model: {model_id}...")
        logger.info(f"Dialogue to summarize: \n{text}")
        
        try:
            client = InferenceClient(api_key=hf_token)
            # Query Hugging Face's serverless summarization task
            result = client.summarization(text, model=model_id)
            
            if isinstance(result, list) and len(result) > 0:
                summary = result[0].get("summary_text", "")
            elif isinstance(result, dict):
                summary = result.get("summary_text", "")
            elif isinstance(result, str):
                summary = result
            else:
                summary = str(result)
            
            # Clean up Pegasus/BART tokenizer newlines
            summary = summary.replace("<n>", "\n")
            logger.info(f"Summary generated successfully: {summary}")
            return summary
        except Exception as e:
            logger.exception(e)
            return f"Error generating summary: {e}"

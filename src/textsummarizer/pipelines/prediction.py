import os
import requests
from textsummarizer.logging import logger

class PredictionPipeline:
    def __init__(self):
        self.use_config = False

    def predict(self, text):
        model_id = os.environ.get("PEGASUS_MODEL_ID", "transformersbook/pegasus-samsum")
        hf_token = os.environ.get("HF_TOKEN")
        
        logger.info(f"Predicting summary via HF Inference API for model: {model_id}...")
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            
        logger.info(f"Dialogue to summarize: \n{text}")
        
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": text})
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    summary = result[0].get("summary_text", "")
                elif isinstance(result, dict):
                    summary = result.get("summary_text", "")
                else:
                    summary = str(result)
                
                # Clean up Pegasus tokenizer newlines
                summary = summary.replace("<n>", "\n")
                logger.info(f"Summary generated successfully: {summary}")
                return summary
            else:
                error_msg = f"HF API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return f"Error generating summary: {error_msg}"
        except Exception as e:
            logger.exception(e)
            return f"Error connecting to HF Inference API: {e}"

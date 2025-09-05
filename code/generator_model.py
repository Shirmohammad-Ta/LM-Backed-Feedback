import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class GeneratorModel:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate(self, prompts, max_length=512, temperature=0.7):
        
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        generated_texts = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        return generated_texts
    
    def get_log_probs(self, texts):
        
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_probs = -outputs.loss * inputs["input_ids"].size(1)
            
        return log_probs

    def parameters(self):
        return self.model.parameters()
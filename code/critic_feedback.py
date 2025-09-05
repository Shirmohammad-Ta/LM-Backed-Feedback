import openai
import torch
import re
from transformers import pipeline, AutoTokenizer

class CriticFeedback:
    def __init__(self, model_name, use_api=False, api_key=None):
        self.use_api = use_api
        
        if use_api and api_key:
            openai.api_key = api_key
        else:
            
            self.critic_model = pipeline(
                "text-generation",
                model=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_feedback(self, texts, task_type="summarization"):
        """Result"""
        if self.use_api:
            return self._get_api_feedback(texts, task_type)
        else:
            return self._get_local_feedback(texts, task_type)
    
    def _get_api_feedback(self, texts, task_type):
        """Result API"""
        rewards = []
        
        for text in texts:
            if task_type == "summarization":
                prompt = f"""
                Rate the quality of this summary (0.0 to 1.0). 
                Consider accuracy, coherence, and conciseness.
                Text: {text}
                Rating: """
            else:  # code generation
                prompt = f"""
                Rate this Python code (0.0 to 1.0). 
                Consider correctness, efficiency, and style.
                Code: {text}
                Rating: """
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0
                )
                
                rating_text = response.choices[0].message.content.strip()
                
                rating = float(re.findall(r"[-+]?\d*\.\d+|\d+", rating_text)[0])
                rewards.append(min(max(rating, 0.0), 1.0))
                
            except Exception as e:
                print(f"API Error: {e}")
                rewards.append(0.5)  
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _get_local_feedback(self, texts, task_type):
        """Recive"""
        
        rewards = []
        for text in texts:
            
            length = len(text.split())
            quality_score = min(length / 100, 1.0)  
            rewards.append(quality_score)
            
        return torch.tensor(rewards, dtype=torch.float32)
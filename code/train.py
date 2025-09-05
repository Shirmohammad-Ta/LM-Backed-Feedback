import yaml
import torch
import logging
from tqdm import tqdm
from datetime import datetime

from generator_model import GeneratorModel
from critic_feedback import CriticFeedback
from policy_optimizer import PolicyOptimizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
   
    config = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    logger.info("Initializing models...")
    generator = GeneratorModel(config['generator_model_name'], device)
    critic = CriticFeedback(
        config['critic_model_name'], 
        config['use_critic_api'],
        config.get('openai_api_key')
    )
    optimizer = PolicyOptimizer(
        generator.model,
        config['learning_rate'],
        config['entropy_coeff']
    )
    
    
    sample_prompts = [
        "Summarize the following text: ...",
        "Write a Python function to ...",
        # ...
    ] * config['batch_size']
    
    
    logger.info("Starting training...")
    for iteration in tqdm(range(config['num_iterations'])):
        try:
            
            generated_texts = generator.generate(
                sample_prompts, 
                config['max_length']
            )
            
            
            rewards = critic.get_feedback(
                generated_texts, 
                "summarization" if "summar" in config['dataset_name'] else "code"
            )
            
            
            log_probs = generator.get_log_probs(generated_texts)
            entropy = torch.ones_like(log_probs) * 0.1  
            
            metrics = optimizer.update(rewards, log_probs, entropy)
            
            
            if iteration % config['log_interval'] == 0:
                tqdm.write(f"Iter {iteration}: Reward={metrics['avg_reward']:.3f}, Loss={metrics['total_loss']:.4f}")
                
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {str(e)}")
            continue
    
    logger.info("Training completed!")
    
    
    torch.save(
        generator.model.state_dict(), 
        f"{config['output_dir']}/final_model.pt"
    )

if __name__ == "__main__":
    main()
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class PolicyOptimizer:
    def __init__(self, model, learning_rate=1e-6, entropy_coeff=0.01):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.entropy_coeff = entropy_coeff
        self.baseline = None
        self.alpha = 0.9  
        
    def update(self, rewards, log_probs, entropy):
        """ policy gradient"""
        
        
        if self.baseline is None:
            self.baseline = rewards.mean()
        else:
            self.baseline = self.alpha * self.baseline + (1 - self.alpha) * rewards.mean()
        
        
        advantages = rewards - self.baseline
        
        
        policy_loss = -(advantages * log_probs).mean()
        entropy_loss = -self.entropy_coeff * entropy.mean()
        total_loss = policy_loss + entropy_loss
        
        
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "avg_reward": rewards.mean().item(),
            "baseline": self.baseline.item()
        }
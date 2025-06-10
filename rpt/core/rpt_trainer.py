"""
RPT Trainer Implementation

This module implements the main training loop for Reinforcement Pre-Training,
integrating reward systems, model training, and scaling utilities.
"""

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

from .reward_system import RewardSystem, AdaptiveRewardSystem
from .rpt_model import RPTModel

logger = logging.getLogger(__name__)


class RPTTrainer:
    """
    Main trainer class for Reinforcement Pre-Training.
    
    Handles the complete training pipeline including:
    - Model forward/backward passes
    - Reward computation and integration
    - Policy gradient optimization
    - Evaluation and logging
    """
    
    def __init__(
        self,
        model: RPTModel,
        reward_system: RewardSystem,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        max_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 100,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        output_dir: str = "./rpt_outputs",
        device: Optional[str] = None,
        mixed_precision: bool = False,
        ppo_epochs: int = 4,
        ppo_clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        """
        Initialize RPT Trainer.
        
        Args:
            model: RPT model to train
            reward_system: Reward computation system
            optimizer: Optimizer for training
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            max_epochs: Maximum training epochs
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            logging_steps: Steps between logging
            eval_steps: Steps between evaluation
            save_steps: Steps between model saves
            output_dir: Directory for outputs
            device: Training device
            mixed_precision: Whether to use mixed precision
            ppo_epochs: Number of PPO epochs per training step
            ppo_clip_param: PPO clipping parameter
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy regularization
        """
        self.model = model
        self.reward_system = reward_system
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        
        # PPO parameters
        self.ppo_epochs = ppo_epochs
        self.ppo_clip_param = ppo_clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Device setup
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Mixed precision
        self.mixed_precision = mixed_precision
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_reward = float('-inf')
        
        # Logging
        self.train_logs = []
        self.eval_logs = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results and statistics
        """
        logger.info(f"Starting RPT training for {self.max_epochs} epochs")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Total training steps: {len(self.train_dataloader) * self.max_epochs}")
        
        self.model.train()
        
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            epoch_logs = self._train_epoch()
            
            # Log epoch results
            avg_reward = np.mean([log["reward"] for log in epoch_logs])
            avg_loss = np.mean([log["loss"] for log in epoch_logs])
            
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs} - Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}")
            
            # Validation
            if self.val_dataloader:
                eval_results = self.evaluate()
                logger.info(f"Validation - Avg Reward: {eval_results['avg_reward']:.4f}")
                
                # Save best model
                if eval_results['avg_reward'] > self.best_eval_reward:
                    self.best_eval_reward = eval_results['avg_reward']
                    self._save_model("best_model")
                    
        # Final save
        self._save_model("final_model")
        
        return {
            "final_step": self.global_step,
            "best_eval_reward": self.best_eval_reward,
            "train_logs": self.train_logs,
            "eval_logs": self.eval_logs
        }
    
    def _train_epoch(self) -> List[Dict[str, float]]:
        """Train for one epoch."""
        epoch_logs = []
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Training step
            step_logs = self._training_step(batch)
            epoch_logs.append(step_logs)
            
            # Update progress bar
            progress_bar.set_postfix({
                "reward": f"{step_logs['reward']:.4f}",
                "loss": f"{step_logs['loss']:.4f}"
            })
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                self._log_training_step(step_logs)
                
            # Evaluation
            if self.global_step % self.eval_steps == 0 and self.val_dataloader:
                eval_results = self.evaluate()
                self._log_evaluation(eval_results)
                
            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint()
                
            self.global_step += 1
            
        return epoch_logs
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step with PPO-style optimization.
        
        Args:
            batch: Training batch
            
        Returns:
            Training step logs
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        
        # Shift for next-token prediction
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, :-1].contiguous()
        
        # Get model outputs
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs["logits"]
            values = outputs.get("values")
            
            # Compute rewards
            rewards = self.reward_system.compute_rewards(
                predictions=logits,
                targets=targets,
                attention_mask=attention_mask
            )
            
            # PPO training
            total_loss = self._compute_ppo_loss(
                logits=logits,
                targets=targets,
                rewards=rewards,
                values=values,
                attention_mask=attention_mask
            )
        
        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(total_loss).backward()
            
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            total_loss.backward()
            
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # Compute statistics
        reward_stats = self.reward_system.compute_reward_statistics(rewards, attention_mask)
        
        # Update adaptive reward system
        if isinstance(self.reward_system, AdaptiveRewardSystem):
            self.reward_system.update_parameters(reward_stats)
        
        return {
            "loss": total_loss.item(),
            "reward": reward_stats["mean_reward"],
            "reward_std": reward_stats["std_reward"],
            "positive_reward_ratio": reward_stats["positive_reward_ratio"]
        }
    
    def _compute_ppo_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        rewards: torch.Tensor,
        values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute PPO loss combining policy and value losses.
        
        Args:
            logits: Model logits
            targets: Target tokens
            rewards: Computed rewards
            values: Value estimates
            attention_mask: Attention mask
            
        Returns:
            Total PPO loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Policy loss (simplified PPO)
        advantages = rewards
        if values is not None:
            # Use value estimates as baseline
            advantages = rewards - values.squeeze(-1)
            
        policy_loss = -target_log_probs * advantages.detach()
        
        # Value loss
        value_loss = torch.zeros_like(policy_loss)
        if values is not None:
            value_targets = rewards
            value_loss = F.mse_loss(values.squeeze(-1), value_targets.detach(), reduction='none')
        
        # Entropy regularization
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_loss = -entropy  # Negative because we want to maximize entropy
        
        # Combine losses
        total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Apply attention mask
        if attention_mask is not None:
            total_loss = total_loss * attention_mask.float()
            
        return total_loss.mean()
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Returns:
            Evaluation results
        """
        if not self.val_dataloader:
            return {}
            
        self.model.eval()
        eval_logs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                
                # Shift for next-token prediction
                targets = input_ids[:, 1:].contiguous()
                inputs = input_ids[:, :-1].contiguous()
                
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :-1].contiguous()
                
                # Get predictions
                outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                logits = outputs["logits"]
                
                # Compute rewards
                rewards = self.reward_system.compute_rewards(
                    predictions=logits,
                    targets=targets,
                    attention_mask=attention_mask
                )
                
                # Compute statistics
                reward_stats = self.reward_system.compute_reward_statistics(rewards, attention_mask)
                eval_logs.append(reward_stats)
        
        self.model.train()
        
        # Aggregate results
        avg_results = {}
        for key in eval_logs[0].keys():
            avg_results[f"avg_{key}"] = np.mean([log[key] for log in eval_logs])
            
        return avg_results
    
    def _log_training_step(self, step_logs: Dict[str, float]) -> None:
        """Log training step results."""
        log_entry = {
            "step": self.global_step,
            "epoch": self.epoch,
            "timestamp": datetime.now().isoformat(),
            **step_logs
        }
        
        self.train_logs.append(log_entry)
        
        # Save logs periodically
        if len(self.train_logs) % 100 == 0:
            self._save_logs()
    
    def _log_evaluation(self, eval_results: Dict[str, float]) -> None:
        """Log evaluation results."""
        log_entry = {
            "step": self.global_step,
            "epoch": self.epoch,
            "timestamp": datetime.now().isoformat(),
            **eval_results
        }
        
        self.eval_logs.append(log_entry)
        self._save_logs()
    
    def _save_model(self, name: str) -> None:
        """Save model checkpoint."""
        save_path = os.path.join(self.output_dir, name)
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def _save_checkpoint(self) -> None:
        """Save full training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_reward": self.best_eval_reward,
            "reward_system_state": self.reward_system.__dict__.copy()
        }
        
        if self.mixed_precision:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{self.global_step}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _save_logs(self) -> None:
        """Save training and evaluation logs."""
        logs = {
            "train_logs": self.train_logs,
            "eval_logs": self.eval_logs
        }
        
        logs_path = os.path.join(self.output_dir, "training_logs.json")
        with open(logs_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_eval_reward = checkpoint["best_eval_reward"]
        
        # Restore reward system state
        reward_system_state = checkpoint.get("reward_system_state", {})
        for key, value in reward_system_state.items():
            if hasattr(self.reward_system, key):
                setattr(self.reward_system, key, value)
        
        if self.mixed_precision and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from step {self.global_step}, epoch {self.epoch}")
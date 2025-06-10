"""
Reward System for Reinforcement Pre-Training

This module implements the reward mechanism for next-token prediction,
treating it as a reasoning task with verifiable rewards.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class RewardSystem:
    """
    Implements verifiable rewards for next-token prediction in RPT.
    
    The reward system evaluates the quality of next-token predictions
    by comparing predicted tokens with ground truth tokens and providing
    scaled rewards based on prediction accuracy and confidence.
    """
    
    def __init__(
        self,
        reward_scale: float = 1.0,
        confidence_threshold: float = 0.5,
        reward_type: str = "accuracy",
        temperature: float = 1.0
    ):
        """
        Initialize the reward system.
        
        Args:
            reward_scale: Scaling factor for rewards
            confidence_threshold: Minimum confidence for positive rewards
            reward_type: Type of reward computation ("accuracy", "confidence", "hybrid")
            temperature: Temperature for confidence calculation
        """
        self.reward_scale = reward_scale
        self.confidence_threshold = confidence_threshold
        self.reward_type = reward_type
        self.temperature = temperature
        
    def compute_rewards(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute rewards for next-token predictions.
        
        Args:
            predictions: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Rewards tensor [batch_size, seq_len]
        """
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Apply softmax to get probabilities
        probs = F.softmax(predictions / self.temperature, dim=-1)
        
        # Get predicted token probabilities
        predicted_probs = torch.gather(probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        if self.reward_type == "accuracy":
            rewards = self._accuracy_rewards(predictions, targets)
        elif self.reward_type == "confidence":
            rewards = self._confidence_rewards(predicted_probs)
        elif self.reward_type == "hybrid":
            acc_rewards = self._accuracy_rewards(predictions, targets)
            conf_rewards = self._confidence_rewards(predicted_probs)
            rewards = 0.5 * acc_rewards + 0.5 * conf_rewards
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        # Apply attention mask if provided
        if attention_mask is not None:
            rewards = rewards * attention_mask.float()
            
        return rewards * self.reward_scale
    
    def _accuracy_rewards(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute accuracy-based rewards."""
        predicted_tokens = torch.argmax(predictions, dim=-1)
        correct_predictions = (predicted_tokens == targets).float()
        return correct_predictions
    
    def _confidence_rewards(self, predicted_probs: torch.Tensor) -> torch.Tensor:
        """Compute confidence-based rewards."""
        # Reward based on prediction confidence
        confidence_rewards = torch.where(
            predicted_probs > self.confidence_threshold,
            predicted_probs,
            torch.zeros_like(predicted_probs)
        )
        return confidence_rewards
    
    def compute_reward_statistics(
        self,
        rewards: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute statistics for reward analysis.
        
        Args:
            rewards: Computed rewards [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with reward statistics
        """
        if attention_mask is not None:
            valid_rewards = rewards[attention_mask.bool()]
        else:
            valid_rewards = rewards.flatten()
            
        stats = {
            "mean_reward": float(valid_rewards.mean()),
            "std_reward": float(valid_rewards.std()),
            "min_reward": float(valid_rewards.min()),
            "max_reward": float(valid_rewards.max()),
            "positive_reward_ratio": float((valid_rewards > 0).float().mean())
        }
        
        return stats
    
    def get_reward_mask(
        self,
        rewards: torch.Tensor,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Get a mask for rewards above a threshold.
        
        Args:
            rewards: Computed rewards
            threshold: Reward threshold (uses confidence_threshold if None)
            
        Returns:
            Boolean mask for valid rewards
        """
        if threshold is None:
            threshold = self.confidence_threshold
            
        return rewards > threshold


class AdaptiveRewardSystem(RewardSystem):
    """
    Adaptive reward system that adjusts reward parameters during training.
    """
    
    def __init__(
        self,
        initial_reward_scale: float = 1.0,
        adaptive_rate: float = 0.01,
        target_positive_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__(reward_scale=initial_reward_scale, **kwargs)
        self.initial_reward_scale = initial_reward_scale
        self.adaptive_rate = adaptive_rate
        self.target_positive_ratio = target_positive_ratio
        self.step_count = 0
        
    def update_parameters(self, reward_stats: Dict[str, float]) -> None:
        """
        Update reward parameters based on training statistics.
        
        Args:
            reward_stats: Statistics from compute_reward_statistics
        """
        current_positive_ratio = reward_stats["positive_reward_ratio"]
        
        # Adapt reward scale to maintain target positive reward ratio
        if current_positive_ratio < self.target_positive_ratio:
            self.reward_scale *= (1 + self.adaptive_rate)
        elif current_positive_ratio > self.target_positive_ratio:
            self.reward_scale *= (1 - self.adaptive_rate)
            
        # Ensure reward scale stays positive
        self.reward_scale = max(0.1, self.reward_scale)
        
        self.step_count += 1
        
    def get_adaptation_info(self) -> Dict[str, float]:
        """Get information about current adaptation state."""
        return {
            "current_reward_scale": self.reward_scale,
            "initial_reward_scale": self.initial_reward_scale,
            "adaptation_steps": self.step_count,
            "scale_ratio": self.reward_scale / self.initial_reward_scale
        }
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
            
        # CRITICAL: Clamp final rewards to prevent loss explosion
        final_rewards = rewards * self.reward_scale
        final_rewards = torch.clamp(final_rewards, -1.0, 1.0)  # Keep rewards reasonable
        return final_rewards
    
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
            
        if valid_rewards.numel() == 0:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "positive_reward_ratio": 0.0
            }
            
        stats = {
            "mean_reward": float(valid_rewards.mean()),
            "std_reward": float(valid_rewards.std()),
            "min_reward": float(valid_rewards.min()),
            "max_reward": float(valid_rewards.max()),
            "positive_reward_ratio": float((valid_rewards > 0).float().mean())
        }
        
        return stats


class MultiObjectiveRewardSystem(RewardSystem):
    """
    Multi-objective reward system combining multiple reward signals.
    Enhances standard rewards with factuality, coherence, relevance, and fluency.
    """
    
    def __init__(
        self,
        factuality_weight: float = 0.25,
        coherence_weight: float = 0.25,
        relevance_weight: float = 0.25,
        fluency_weight: float = 0.25,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.factuality_weight = factuality_weight
        self.coherence_weight = coherence_weight
        self.relevance_weight = relevance_weight
        self.fluency_weight = fluency_weight
        
        # Normalize weights
        total_weight = sum([factuality_weight, coherence_weight, relevance_weight, fluency_weight])
        self.factuality_weight /= total_weight
        self.coherence_weight /= total_weight
        self.relevance_weight /= total_weight
        self.fluency_weight /= total_weight
        
    def compute_factuality_reward(self, predictions, targets):
        """Compute factual consistency reward using simple heuristics"""
        batch_size, seq_len = targets.shape
        
        # Simple factuality heuristic: reward consistency in numerical predictions
        pred_tokens = torch.argmax(predictions, dim=-1)
        
        # Create random baseline rewards for demo (replace with learned factuality model)
        factuality_scores = torch.rand(batch_size, seq_len, device=predictions.device)
        
        # Boost reward for exact matches (indicating potential factual accuracy)
        exact_matches = (pred_tokens == targets).float()
        factuality_scores = factuality_scores * 0.7 + exact_matches * 0.3
        
        return factuality_scores
        
    def compute_coherence_reward(self, predictions):
        """Compute coherence reward based on prediction entropy"""
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Use entropy as coherence proxy - lower entropy = more coherent
        probs = F.softmax(predictions, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float))
        
        # Invert entropy to reward coherence (lower entropy = higher reward)
        coherence_scores = 1.0 - (entropy / max_entropy)
        coherence_scores = torch.clamp(coherence_scores, 0.0, 1.0)
        
        return coherence_scores
        
    def compute_relevance_reward(self, predictions, targets):
        """Compute relevance reward based on target alignment"""
        batch_size, seq_len = targets.shape
        
        # Use target probability as relevance proxy
        probs = F.softmax(predictions, dim=-1)
        target_probs = torch.gather(probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Scale to [0,1] range
        relevance_scores = torch.clamp(target_probs * 2.0, 0.0, 1.0)
        
        return relevance_scores
        
    def compute_fluency_reward(self, predictions):
        """Compute fluency reward based on prediction confidence"""
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Use maximum probability as fluency indicator
        probs = F.softmax(predictions, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        
        # Apply sigmoid to get smoother rewards
        fluency_scores = torch.sigmoid(max_probs * 6 - 3)  # Center around 0.5
        
        return fluency_scores
        
    def compute_rewards(self, predictions, targets, attention_mask=None):
        """Compute multi-objective rewards combining all aspects"""
        # Get base rewards
        base_rewards = super().compute_rewards(predictions, targets, attention_mask)
        
        # Compute specialized rewards
        factuality = self.compute_factuality_reward(predictions, targets)
        coherence = self.compute_coherence_reward(predictions)
        relevance = self.compute_relevance_reward(predictions, targets)
        fluency = self.compute_fluency_reward(predictions)
        
        # Combine with learned weights
        combined_rewards = (
            base_rewards * 0.4 +  # Keep some base reward
            factuality * self.factuality_weight +
            coherence * self.coherence_weight +
            relevance * self.relevance_weight +
            fluency * self.fluency_weight
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            combined_rewards = combined_rewards * attention_mask.float()
            
        return combined_rewards * self.reward_scale


class HumanFeedbackRewardSystem(RewardSystem):
    """
    Human feedback integration for RLHF-style training.
    """
    
    def __init__(self, feedback_weight: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.feedback_weight = feedback_weight
        self.feedback_buffer = []
        
    def add_human_feedback(self, input_text: str, output_text: str, score: float):
        """Add human feedback to the system"""
        self.feedback_buffer.append({
            'input': input_text,
            'output': output_text, 
            'score': score,
            'timestamp': torch.tensor(len(self.feedback_buffer))
        })
        
    def compute_feedback_reward(self, predictions, context=None):
        """Compute rewards based on stored human feedback"""
        batch_size, seq_len = predictions.shape[:2]
        
        if len(self.feedback_buffer) == 0:
            return torch.zeros(batch_size, seq_len, device=predictions.device)
            
        # Simple demo: use average feedback score
        avg_feedback = sum(fb['score'] for fb in self.feedback_buffer) / len(self.feedback_buffer)
        feedback_rewards = torch.full(
            (batch_size, seq_len), 
            avg_feedback, 
            device=predictions.device
        )
        
        return feedback_rewards
        
    def compute_rewards(self, predictions, targets, attention_mask=None):
        """Combine standard rewards with human feedback"""
        base_rewards = super().compute_rewards(predictions, targets, attention_mask)
        feedback_rewards = self.compute_feedback_reward(predictions)
        
        combined_rewards = (
            base_rewards * (1 - self.feedback_weight) +
            feedback_rewards * self.feedback_weight
        )
        
        if attention_mask is not None:
            combined_rewards = combined_rewards * attention_mask.float()
            
        return combined_rewards
    
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
"""
Metrics and Evaluation for RPT

This module provides comprehensive metrics for evaluating
Reinforcement Pre-Training performance and next-token reasoning.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RPTMetrics:
    """
    Comprehensive metrics suite for evaluating RPT training and performance.
    """
    
    def __init__(self, track_detailed_stats: bool = True):
        """
        Initialize RPT metrics tracker.
        
        Args:
            track_detailed_stats: Whether to track detailed statistics
        """
        self.track_detailed_stats = track_detailed_stats
        self.reset()
        
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics_history = defaultdict(list)
        self.detailed_stats = defaultdict(list) if self.track_detailed_stats else None
        
    def compute_token_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute token-level accuracy metrics.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Token accuracy metrics
        """
        # Get predicted tokens
        predicted_tokens = torch.argmax(predictions, dim=-1)
        
        # Compute accuracy
        correct_predictions = (predicted_tokens == targets).float()
        
        if attention_mask is not None:
            # Only count valid positions
            valid_mask = attention_mask.float()
            total_correct = (correct_predictions * valid_mask).sum()
            total_valid = valid_mask.sum()
            accuracy = total_correct / (total_valid + 1e-8)
            
            # Per-position accuracy
            position_accuracy = (correct_predictions * valid_mask).sum(dim=0) / (valid_mask.sum(dim=0) + 1e-8)
        else:
            accuracy = correct_predictions.mean()
            position_accuracy = correct_predictions.mean(dim=0)
        
        return {
            "token_accuracy": float(accuracy),
            "position_accuracy": position_accuracy.cpu().numpy().tolist(),
            "total_tokens": int(total_valid) if attention_mask is not None else predictions.numel()
        }
    
    def compute_perplexity(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute perplexity metrics.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Perplexity metrics
        """
        # Compute log probabilities
        log_probs = torch.log_softmax(predictions, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        if attention_mask is not None:
            # Mask invalid positions
            target_log_probs = target_log_probs * attention_mask.float()
            total_tokens = attention_mask.sum()
            avg_log_prob = target_log_probs.sum() / (total_tokens + 1e-8)
        else:
            avg_log_prob = target_log_probs.mean()
            total_tokens = target_log_probs.numel()
        
        perplexity = torch.exp(-avg_log_prob)
        
        return {
            "perplexity": float(perplexity),
            "log_perplexity": float(-avg_log_prob),
            "total_tokens": int(total_tokens)
        }
    
    def compute_reasoning_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute metrics specific to reasoning quality in RPT.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            rewards: Computed rewards [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Reasoning quality metrics
        """
        # Prediction confidence
        probs = torch.softmax(predictions, dim=-1)
        predicted_probs = torch.gather(probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Entropy (uncertainty)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        if attention_mask is not None:
            mask = attention_mask.float()
            valid_confidence = (predicted_probs * mask).sum() / (mask.sum() + 1e-8)
            valid_entropy = (entropy * mask).sum() / (mask.sum() + 1e-8)
            valid_rewards = (rewards * mask).sum() / (mask.sum() + 1e-8)
        else:
            valid_confidence = predicted_probs.mean()
            valid_entropy = entropy.mean()
            valid_rewards = rewards.mean()
        
        # High-confidence predictions
        high_confidence_threshold = 0.8
        high_confidence_mask = predicted_probs > high_confidence_threshold
        if attention_mask is not None:
            high_confidence_mask = high_confidence_mask & attention_mask.bool()
        
        high_confidence_ratio = high_confidence_mask.float().mean()
        
        # Reward-confidence correlation
        if attention_mask is not None:
            valid_indices = attention_mask.bool()
            conf_flat = predicted_probs[valid_indices]
            reward_flat = rewards[valid_indices]
        else:
            conf_flat = predicted_probs.flatten()
            reward_flat = rewards.flatten()
        
        if len(conf_flat) > 1:
            correlation = torch.corrcoef(torch.stack([conf_flat, reward_flat]))[0, 1]
            correlation = correlation if not torch.isnan(correlation) else 0.0
        else:
            correlation = 0.0
        
        return {
            "avg_confidence": float(valid_confidence),
            "avg_entropy": float(valid_entropy),
            "avg_reward": float(valid_rewards),
            "high_confidence_ratio": float(high_confidence_ratio),
            "reward_confidence_correlation": float(correlation)
        }
    
    def compute_scaling_metrics(
        self,
        model: torch.nn.Module,
        batch_size: int,
        sequence_length: int,
        training_time: float
    ) -> Dict[str, float]:
        """
        Compute metrics related to training scaling and efficiency.
        
        Args:
            model: The model being trained
            batch_size: Current batch size
            sequence_length: Sequence length
            training_time: Time taken for this step
            
        Returns:
            Scaling metrics
        """
        # Model size metrics
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Memory usage (if CUDA available)
        memory_allocated = 0
        memory_reserved = 0
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        
        # Throughput metrics
        tokens_processed = batch_size * sequence_length
        tokens_per_second = tokens_processed / (training_time + 1e-8)
        
        return {
            "param_count": param_count,
            "trainable_params": trainable_params,
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
            "tokens_per_second": tokens_per_second,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "training_time": training_time
        }
    
    def update_metrics(
        self,
        step: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Update all metrics for a training step.
        
        Args:
            step: Current training step
            predictions: Model predictions
            targets: Target tokens
            rewards: Computed rewards
            attention_mask: Attention mask
            additional_metrics: Additional metrics to track
            
        Returns:
            Combined metrics for this step
        """
        # Compute all metric types
        accuracy_metrics = self.compute_token_accuracy(predictions, targets, attention_mask)
        perplexity_metrics = self.compute_perplexity(predictions, targets, attention_mask)
        reasoning_metrics = self.compute_reasoning_metrics(predictions, targets, rewards, attention_mask)
        
        # Combine all metrics
        all_metrics = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **accuracy_metrics,
            **perplexity_metrics,
            **reasoning_metrics
        }
        
        if additional_metrics:
            all_metrics.update(additional_metrics)
        
        # Update history
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
        
        # Track detailed stats if enabled
        if self.track_detailed_stats:
            self.detailed_stats[step] = all_metrics
        
        return all_metrics
    
    def get_summary_stats(self, window_size: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for recent metrics.
        
        Args:
            window_size: Number of recent steps to summarize
            
        Returns:
            Summary statistics
        """
        summary = {}
        
        for metric_name, values in self.metrics_history.items():
            if len(values) == 0:
                continue
                
            recent_values = values[-window_size:] if len(values) > window_size else values
            
            if isinstance(recent_values[0], (int, float)):
                summary[metric_name] = {
                    "mean": float(np.mean(recent_values)),
                    "std": float(np.std(recent_values)),
                    "min": float(np.min(recent_values)),
                    "max": float(np.max(recent_values)),
                    "latest": float(recent_values[-1]),
                    "trend": self._compute_trend(recent_values)
                }
        
        return summary
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction for a metric."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def plot_metrics(
        self,
        metrics_to_plot: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot training metrics over time.
        
        Args:
            metrics_to_plot: List of metrics to plot (None for all)
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.metrics_history:
            logger.warning("No metrics to plot")
            return
        
        # Default metrics to plot
        if metrics_to_plot is None:
            metrics_to_plot = [
                "token_accuracy", "perplexity", "avg_reward", 
                "avg_confidence", "avg_entropy"
            ]
        
        # Filter available metrics
        available_metrics = [m for m in metrics_to_plot if m in self.metrics_history and len(self.metrics_history[m]) > 0]
        
        if not available_metrics:
            logger.warning("No available metrics to plot")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(available_metrics):
            values = self.metrics_history[metric]
            steps = list(range(len(values)))
            
            axes[i].plot(steps, values)
            axes[i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics plot saved to {save_path}")
        
        plt.show()
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to save metrics
        """
        export_data = {
            "metrics_history": dict(self.metrics_history),
            "summary_stats": self.get_summary_stats(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        if self.detailed_stats:
            export_data["detailed_stats"] = dict(self.detailed_stats)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def load_metrics(self, filepath: str) -> None:
        """
        Load metrics from JSON file.
        
        Args:
            filepath: Path to load metrics from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load metrics history
        for key, values in data.get("metrics_history", {}).items():
            self.metrics_history[key] = values
        
        # Load detailed stats if available
        if "detailed_stats" in data and self.track_detailed_stats:
            for step, stats in data["detailed_stats"].items():
                self.detailed_stats[int(step)] = stats
        
        logger.info(f"Metrics loaded from {filepath}")
    
    def compare_experiments(
        self,
        other_metrics: 'RPTMetrics',
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics with another experiment.
        
        Args:
            other_metrics: Other RPTMetrics instance to compare with
            metric_names: Specific metrics to compare
            
        Returns:
            Comparison results
        """
        if metric_names is None:
            metric_names = list(set(self.metrics_history.keys()) & set(other_metrics.metrics_history.keys()))
        
        comparison = {}
        
        for metric in metric_names:
            if metric in self.metrics_history and metric in other_metrics.metrics_history:
                self_values = self.metrics_history[metric]
                other_values = other_metrics.metrics_history[metric]
                
                if self_values and other_values:
                    comparison[metric] = {
                        "self_final": float(self_values[-1]),
                        "other_final": float(other_values[-1]),
                        "self_mean": float(np.mean(self_values)),
                        "other_mean": float(np.mean(other_values)),
                        "improvement": float(self_values[-1] - other_values[-1]),
                        "relative_improvement": float((self_values[-1] - other_values[-1]) / (other_values[-1] + 1e-8))
                    }
        
        return comparison
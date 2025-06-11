"""
RPT Model Implementation

This module provides a wrapper for language models to enable
Reinforcement Pre-Training with next-token reasoning capabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import logging

logger = logging.getLogger(__name__)


class RPTModel(nn.Module):
    """
    RPT Model wrapper that integrates language models with RL training.
    
    This class wraps existing language models and adds components necessary
    for reinforcement pre-training, including value heads and action spaces.
    """
    
    def __init__(
        self,
        base_model: Union[str, PreTrainedModel],
        tokenizer: Optional[Any] = None,
        add_value_head: bool = True,
        value_head_dim: int = 1,
        freeze_base_model: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize RPT Model.
        
        Args:
            base_model: Base language model (model name or instance)
            tokenizer: Tokenizer for the model
            add_value_head: Whether to add a value estimation head
            value_head_dim: Dimension of value head output
            freeze_base_model: Whether to freeze base model parameters
            device: Device to place model on
        """
        super().__init__()
        
        # Load base model
        if isinstance(base_model, str):
            self.base_model = AutoModel.from_pretrained(base_model)
            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            else:
                self.tokenizer = tokenizer
        else:
            self.base_model = base_model
            self.tokenizer = tokenizer
            
        # Get model dimensions
        self.hidden_size = self.base_model.config.hidden_size
        self.vocab_size = getattr(self.base_model.config, 'vocab_size', 50257)
        
        # Language modeling head (if not present)
        if not hasattr(self.base_model, 'lm_head'):
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        else:
            self.lm_head = self.base_model.lm_head
            
        # Value head for RL training (FIXED: proper initialization)
        if add_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, value_head_dim)
            )
            # CRITICAL: Initialize value head to output small values near 0
            with torch.no_grad():
                for module in self.value_head:
                    if isinstance(module, nn.Linear):
                        # Small weight initialization
                        module.weight.normal_(0, 0.01)
                        if module.bias is not None:
                            module.bias.fill_(0.0)
        else:
            self.value_head = None
            
        # Freeze base model if requested
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # Move to device
        if device:
            self.to(device)
            
        self.device = device or next(self.parameters()).device
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RPT model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return a dictionary
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Dictionary containing logits, values, and optional hidden states
        """
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get hidden states
        hidden_states = base_outputs.last_hidden_state
        
        # Compute language modeling logits
        logits = self.lm_head(hidden_states)
        
        # Compute values if value head exists
        values = None
        if self.value_head is not None:
            values = self.value_head(hidden_states)
            
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states
        }
        
        if values is not None:
            outputs["values"] = values
            
        if output_hidden_states:
            outputs["all_hidden_states"] = base_outputs.hidden_states
            
        return outputs
    
    def generate_with_reasoning(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        reasoning_steps: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text with reasoning-based token selection.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            attention_mask: Attention mask
            reasoning_steps: Number of reasoning steps per token
            
        Returns:
            Dictionary with generated tokens and reasoning information
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        generated = input_ids.clone()
        reasoning_scores = []
        
        with torch.no_grad():
            for step in range(max_length - seq_len):
                # Current sequence
                current_seq = generated
                current_attention_mask = attention_mask
                
                # Multi-step reasoning for next token
                best_logits = None
                best_reasoning_score = float('-inf')
                
                for reasoning_step in range(reasoning_steps):
                    # Forward pass
                    outputs = self.forward(
                        input_ids=current_seq,
                        attention_mask=current_attention_mask
                    )
                    
                    logits = outputs["logits"][:, -1, :]  # Last position logits
                    
                    # Compute reasoning score (using value head if available)
                    if self.value_head is not None:
                        values = outputs["values"][:, -1, :]
                        reasoning_score = values.mean().item()
                    else:
                        # Use entropy as reasoning proxy
                        probs = torch.softmax(logits / temperature, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                        reasoning_score = -entropy.mean().item()  # Lower entropy = better reasoning
                    
                    if reasoning_score > best_reasoning_score:
                        best_reasoning_score = reasoning_score
                        best_logits = logits.clone()
                
                # Select next token based on best reasoning
                if do_sample:
                    probs = torch.softmax(best_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(best_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                reasoning_scores.append(best_reasoning_score)
                
                # Update attention mask
                if current_attention_mask is not None:
                    new_mask = torch.ones(batch_size, 1, device=device)
                    attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)
                    
        return {
            "generated_ids": generated,
            "reasoning_scores": torch.tensor(reasoning_scores),
            "input_length": seq_len
        }
    
    def get_token_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get token probabilities for input sequence.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Token probabilities [batch_size, seq_len, vocab_size]
        """
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return torch.softmax(outputs["logits"], dim=-1)
    
    def estimate_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Estimate values for input sequences.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Value estimates if value head exists, None otherwise
        """
        if self.value_head is None:
            return None
            
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["values"]
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save the RPT model."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(os.path.join(save_directory, "base_model"))
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        
        # Save additional components
        additional_state = {
            "lm_head": self.lm_head.state_dict() if hasattr(self, 'lm_head') else None,
            "value_head": self.value_head.state_dict() if self.value_head else None,
            "config": {
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size,
                "has_value_head": self.value_head is not None
            }
        }
        
        torch.save(additional_state, os.path.join(save_directory, "rpt_components.pt"))
        
    @classmethod
    def from_pretrained(cls, load_directory: str, device: Optional[str] = None):
        """Load a pre-trained RPT model."""
        import os
        from transformers import AutoModel, AutoTokenizer
        
        # Load base model
        base_model = AutoModel.from_pretrained(os.path.join(load_directory, "base_model"))
        
        # Load tokenizer
        tokenizer_path = os.path.join(load_directory, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if os.path.exists(tokenizer_path) else None
        
        # Load additional components
        components_path = os.path.join(load_directory, "rpt_components.pt")
        if os.path.exists(components_path):
            additional_state = torch.load(components_path, map_location="cpu")
            config = additional_state["config"]
            
            # Create model
            model = cls(
                base_model=base_model,
                tokenizer=tokenizer,
                add_value_head=config["has_value_head"],
                device=device
            )
            
            # Load additional component states
            if additional_state["lm_head"]:
                model.lm_head.load_state_dict(additional_state["lm_head"])
            if additional_state["value_head"] and model.value_head:
                model.value_head.load_state_dict(additional_state["value_head"])
                
        else:
            model = cls(base_model=base_model, tokenizer=tokenizer, device=device)
            
        return model
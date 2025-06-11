"""
Advanced Model Architectures for Reinforcement Pre-Training

Implements enhanced model architectures including Mixture of Experts,
Memory-Augmented Networks, and Hierarchical Reasoning modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Dict, Any, List, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)


class MixtureOfExpertsLayer(nn.Module):
    """
    Mixture of Experts layer for specialized reasoning capabilities.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        num_active_experts: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gating network to select experts
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim)
            )
            for _ in range(num_experts)
        ])
        
        # Load balancing
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through mixture of experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            output: Mixed expert outputs
            aux_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)
        
        # Compute gating scores
        gate_scores = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_probs, self.num_active_experts, dim=-1)
        top_k_probs = F.softmax(top_k_scores, dim=-1)
        
        # Track expert usage for load balancing
        for i in range(self.num_experts):
            usage = (top_k_indices == i).float().mean()
            self.expert_usage[i] = 0.9 * self.expert_usage[i] + 0.1 * usage
        
        # Compute expert outputs
        outputs = torch.zeros_like(x_flat)
        
        for i in range(self.num_active_experts):
            expert_idx = top_k_indices[:, i]
            expert_prob = top_k_probs[:, i]
            
            # Get unique expert indices for efficient computation
            unique_experts = torch.unique(expert_idx)
            
            for exp_id in unique_experts:
                mask = (expert_idx == exp_id)
                if mask.sum() > 0:
                    expert_input = x_flat[mask]
                    expert_output = self.experts[exp_id](expert_input)
                    expert_weight = expert_prob[mask].unsqueeze(-1)
                    outputs[mask] += expert_weight * expert_output
        
        # Reshape back
        outputs = outputs.view(batch_size, seq_len, input_dim)
        
        # Compute auxiliary loss for load balancing
        importance = gate_probs.mean(0)
        load = (gate_probs > 0).float().mean(0)
        aux_loss = (importance * load).sum() * self.num_experts
        
        return outputs, aux_loss


class MemoryAugmentedAttention(nn.Module):
    """
    Memory-augmented attention mechanism for enhanced reasoning.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        memory_size: int = 256,
        memory_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Memory components
        self.memory_keys = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_proj = nn.Linear(memory_dim, hidden_size)
        
        # Memory attention
        self.memory_attn = nn.Linear(hidden_size, memory_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with memory-augmented attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Enhanced hidden states with memory augmentation
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Standard multi-head attention
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attention_mask is not None:
            attn_scores += attention_mask.unsqueeze(1).unsqueeze(1) * -1e9
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        # Memory augmentation
        memory_attn_scores = self.memory_attn(hidden_states)  # [batch_size, seq_len, memory_size]
        memory_attn_probs = F.softmax(memory_attn_scores, dim=-1)
        
        # Retrieve from memory
        memory_output = torch.matmul(memory_attn_probs, self.memory_values)  # [batch_size, seq_len, memory_dim]
        memory_output = self.memory_proj(memory_output)  # [batch_size, seq_len, hidden_size]
        
        # Combine attention output with memory
        combined_output = attn_output + 0.3 * memory_output
        
        return self.out_proj(combined_output)


class HierarchicalReasoningModule(nn.Module):
    """
    Hierarchical reasoning module for multi-step reasoning tasks.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_reasoning_steps: int = 3,
        reasoning_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_reasoning_steps = num_reasoning_steps
        self.reasoning_dim = reasoning_dim
        
        # Reasoning step modules
        self.reasoning_steps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, reasoning_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(reasoning_dim, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            for _ in range(num_reasoning_steps)
        ])
        
        # Step gating mechanism
        self.step_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )
            for _ in range(num_reasoning_steps)
        ])
        
        # Final integration
        self.integration = nn.Linear(hidden_size * num_reasoning_steps, hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical reasoning.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Dictionary containing final output and step-wise reasoning
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        step_outputs = []
        step_scores = []
        current_state = hidden_states
        
        for i, (reasoning_step, gate) in enumerate(zip(self.reasoning_steps, self.step_gates)):
            # Apply reasoning step
            step_output = reasoning_step(current_state)
            
            # Compute gating score
            gate_score = gate(current_state)
            step_scores.append(gate_score)
            
            # Gated residual connection
            current_state = current_state + gate_score * step_output
            step_outputs.append(current_state)
        
        # Integrate all reasoning steps
        concatenated = torch.cat(step_outputs, dim=-1)
        final_output = self.integration(concatenated)
        
        return {
            'output': final_output,
            'step_outputs': step_outputs,
            'step_scores': step_scores,
            'reasoning_depth': len(step_outputs)
        }


class EnhancedRPTModel(nn.Module):
    """
    Enhanced RPT model with advanced architectural components.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        use_moe: bool = True,
        use_memory: bool = True,
        use_hierarchical: bool = True,
        num_experts: int = 8,
        memory_size: int = 256,
        num_reasoning_steps: int = 3,
        **kwargs
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.use_moe = use_moe
        self.use_memory = use_memory
        self.use_hierarchical = use_hierarchical
        
        # Enhanced components
        if use_moe:
            self.moe_layer = MixtureOfExpertsLayer(
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size * 2,
                num_experts=num_experts
            )
            
        if use_memory:
            self.memory_attention = MemoryAugmentedAttention(
                hidden_size=self.hidden_size,
                memory_size=memory_size
            )
            
        if use_hierarchical:
            self.hierarchical_reasoning = HierarchicalReasoningModule(
                hidden_size=self.hidden_size,
                num_reasoning_steps=num_reasoning_steps
            )
        
        # Value head for RL training
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Reasoning scorer
        self.reasoning_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_reasoning_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with advanced reasoning components.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for training
            return_reasoning_info: Whether to return detailed reasoning info
            
        Returns:
            Dictionary containing model outputs and reasoning information
        """
        # Base model forward pass
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = base_outputs.last_hidden_state
        logits = base_outputs.logits
        
        aux_losses = []
        reasoning_info = {}
        
        # Apply Mixture of Experts
        if self.use_moe:
            moe_output, moe_aux_loss = self.moe_layer(hidden_states)
            hidden_states = hidden_states + 0.5 * moe_output
            aux_losses.append(moe_aux_loss)
            reasoning_info['moe_aux_loss'] = moe_aux_loss
        
        # Apply Memory-Augmented Attention
        if self.use_memory:
            memory_output = self.memory_attention(hidden_states, attention_mask)
            hidden_states = hidden_states + 0.3 * memory_output
            reasoning_info['memory_enhanced'] = True
        
        # Apply Hierarchical Reasoning
        if self.use_hierarchical:
            hierarchical_output = self.hierarchical_reasoning(hidden_states)
            hidden_states = hierarchical_output['output']
            reasoning_info.update({
                'reasoning_steps': hierarchical_output['step_outputs'],
                'reasoning_scores': hierarchical_output['step_scores'],
                'reasoning_depth': hierarchical_output['reasoning_depth']
            })
        
        # Update logits with enhanced hidden states
        enhanced_logits = self.base_model.lm_head(hidden_states)
        
        # Compute values for RL training
        values = self.value_head(hidden_states)
        
        # Compute reasoning scores
        reasoning_scores = self.reasoning_scorer(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = enhanced_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add auxiliary losses
            if aux_losses:
                total_aux_loss = sum(aux_losses)
                loss = loss + 0.01 * total_aux_loss
        
        outputs = {
            'logits': enhanced_logits,
            'values': values,
            'reasoning_scores': reasoning_scores,
            'loss': loss,
            'hidden_states': hidden_states
        }
        
        if return_reasoning_info:
            outputs['reasoning_info'] = reasoning_info
            
        return outputs
    
    def generate_with_enhanced_reasoning(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        reasoning_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text with enhanced reasoning capabilities.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            attention_mask: Attention mask
            reasoning_threshold: Threshold for reasoning-based token selection
            
        Returns:
            Generated tokens and reasoning information
        """
        batch_size, input_length = input_ids.shape
        device = input_ids.device
        
        # Initialize generation
        generated_ids = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        reasoning_scores_history = []
        value_history = []
        
        self.eval()
        with torch.no_grad():
            for step in range(max_length - input_length):
                # Forward pass
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    return_reasoning_info=True
                )
                
                # Get next token logits
                next_token_logits = outputs['logits'][:, -1, :]
                reasoning_scores = outputs['reasoning_scores'][:, -1, :]
                values = outputs['values'][:, -1, :]
                
                # Store reasoning information
                reasoning_scores_history.append(reasoning_scores)
                value_history.append(values)
                
                # Enhanced token selection with reasoning
                if reasoning_scores.mean() > reasoning_threshold:
                    # Use reasoning-enhanced selection
                    enhanced_logits = next_token_logits + reasoning_scores.squeeze(-1) * temperature
                else:
                    enhanced_logits = next_token_logits
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(enhanced_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(enhanced_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, 1, device=device)
                ], dim=1)
                
                # Check for end of sequence
                if next_token.item() == self.base_model.config.eos_token_id:
                    break
        
        return {
            'generated_ids': generated_ids,
            'reasoning_scores': torch.stack(reasoning_scores_history, dim=1) if reasoning_scores_history else None,
            'values': torch.stack(value_history, dim=1) if value_history else None,
            'input_length': input_length
        }
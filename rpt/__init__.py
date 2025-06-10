"""
Reinforcement Pre-Training (RPT) Package

A Python package implementing Reinforcement Pre-Training techniques for language models,
based on the paper "Reinforcement Pre-Training (RPT)" (arXiv:2506.08007).

This package provides tools to:
- Train language models using reinforcement learning
- Treat next-token prediction as reasoning tasks with verifiable rewards
- Scale RL training for large language models
- Evaluate next-token reasoning performance

Main Components:
- RPTTrainer: Main trainer class for reinforcement pre-training
- RewardSystem: Handles reward computation for next-token prediction
- RPTModel: Wrapper for language models with RL integration
- ScalingUtils: Tools for computational scaling
- Metrics: Evaluation and monitoring tools
"""

from .core.rpt_trainer import RPTTrainer
from .core.reward_system import RewardSystem
from .core.rpt_model import RPTModel
from .utils.scaling import ScalingUtils
from .utils.metrics import RPTMetrics
from .utils.data_utils import DataProcessor

__version__ = "0.1.0"
__author__ = "RPT Package Authors"
__description__ = "Reinforcement Pre-Training for Language Models"

__all__ = [
    "RPTTrainer",
    "RewardSystem", 
    "RPTModel",
    "ScalingUtils",
    "RPTMetrics",
    "DataProcessor"
]
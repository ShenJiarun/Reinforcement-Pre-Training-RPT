"""
Curriculum Learning Framework for Reinforcement Pre-Training

Implements progressive learning strategies where models start with simple
tasks and gradually advance to more complex reasoning challenges.
"""

import torch
from typing import List, Dict, Any, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Enumeration of curriculum difficulty levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class CurriculumStage:
    """Defines a stage in the curriculum learning process"""
    name: str
    difficulty: DifficultyLevel
    data: List[str]
    epochs: int = 1
    learning_rate: float = 5e-5
    batch_size: int = 2
    success_threshold: float = 0.7
    description: str = ""


class CurriculumManager:
    """
    Manages curriculum learning progression for RPT training.
    
    Automatically advances through difficulty levels based on model performance
    and provides adaptive learning rates and batch sizes.
    """
    
    def __init__(
        self,
        stages: List[CurriculumStage],
        advancement_patience: int = 3,
        regression_threshold: float = 0.5
    ):
        """
        Initialize curriculum manager.
        
        Args:
            stages: List of curriculum stages in order
            advancement_patience: Epochs to wait before advancing
            regression_threshold: Performance threshold to trigger regression
        """
        self.stages = stages
        self.current_stage_idx = 0
        self.advancement_patience = advancement_patience
        self.regression_threshold = regression_threshold
        
        self.stage_history = []
        self.performance_history = []
        
    @property
    def current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage"""
        return self.stages[self.current_stage_idx]
    
    @property
    def is_final_stage(self) -> bool:
        """Check if we're at the final stage"""
        return self.current_stage_idx >= len(self.stages) - 1
    
    def should_advance(self, recent_performance: List[float]) -> bool:
        """
        Determine if model should advance to next stage.
        
        Args:
            recent_performance: List of recent performance scores
            
        Returns:
            True if should advance to next difficulty level
        """
        if self.is_final_stage or len(recent_performance) < self.advancement_patience:
            return False
            
        # Check if recent performance meets threshold
        avg_performance = sum(recent_performance[-self.advancement_patience:]) / self.advancement_patience
        
        return avg_performance >= self.current_stage.success_threshold
    
    def should_regress(self, recent_performance: List[float]) -> bool:
        """
        Determine if model should regress to easier stage.
        
        Args:
            recent_performance: List of recent performance scores
            
        Returns:
            True if should regress to easier difficulty level
        """
        if self.current_stage_idx == 0 or len(recent_performance) < self.advancement_patience:
            return False
            
        # Check if performance has degraded significantly
        avg_performance = sum(recent_performance[-self.advancement_patience:]) / self.advancement_patience
        
        return avg_performance < self.regression_threshold
    
    def advance_stage(self) -> bool:
        """
        Advance to the next curriculum stage.
        
        Returns:
            True if advancement was successful
        """
        if self.is_final_stage:
            logger.warning("Already at final curriculum stage")
            return False
            
        self.stage_history.append(self.current_stage_idx)
        self.current_stage_idx += 1
        
        logger.info(f"🎓 Advanced to curriculum stage: {self.current_stage.name}")
        logger.info(f"   Difficulty: {self.current_stage.difficulty.value}")
        logger.info(f"   Data samples: {len(self.current_stage.data)}")
        
        return True
    
    def regress_stage(self) -> bool:
        """
        Regress to the previous curriculum stage.
        
        Returns:
            True if regression was successful
        """
        if self.current_stage_idx == 0:
            logger.warning("Already at first curriculum stage")
            return False
            
        self.current_stage_idx -= 1
        
        logger.info(f"📉 Regressed to curriculum stage: {self.current_stage.name}")
        logger.info(f"   Need to improve performance before advancing")
        
        return True
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for current stage"""
        stage = self.current_stage
        return {
            'learning_rate': stage.learning_rate,
            'batch_size': stage.batch_size,
            'epochs': stage.epochs,
            'data': stage.data,
            'stage_name': stage.name,
            'difficulty': stage.difficulty.value
        }
    
    def update_performance(self, performance_score: float):
        """Update performance history"""
        self.performance_history.append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]


def create_default_curriculum() -> List[CurriculumStage]:
    """Create a default curriculum for general conversation training"""
    
    return [
        CurriculumStage(
            name="Basic Facts",
            difficulty=DifficultyLevel.BASIC,
            data=[
                "Q: What is 2 + 2? A: 4",
                "Q: What color is the sky? A: Blue",
                "Q: How many days in a week? A: 7",
                "Human: Hello A: Hello!",
                "Q: What is the capital of France? A: Paris",
            ],
            epochs=2,
            learning_rate=1e-4,
            batch_size=1,
            success_threshold=0.8,
            description="Simple factual questions and basic greetings"
        ),
        
        CurriculumStage(
            name="Simple Reasoning",
            difficulty=DifficultyLevel.INTERMEDIATE,
            data=[
                "Q: If I have 5 apples and eat 2, how many remain? A: 5 - 2 = 3 apples remain",
                "Q: What comes after Monday? A: Tuesday comes after Monday",
                "Q: If it's 3 PM, what time was it 2 hours ago? A: 3 PM - 2 hours = 1 PM",
                "Human: I'm sad. Assistant: I'm sorry to hear you're feeling sad. Would you like to talk about what's bothering you?",
                "Q: Which is bigger: 15 or 12? A: 15 is bigger than 12",
            ],
            epochs=2,
            learning_rate=7e-5,
            batch_size=2,
            success_threshold=0.7,
            description="Basic arithmetic and logical reasoning"
        ),
        
        CurriculumStage(
            name="Complex Reasoning",
            difficulty=DifficultyLevel.ADVANCED,
            data=[
                "Q: A train travels 60 mph for 3 hours. How far does it go? A: Using distance = speed × time: 60 mph × 3 hours = 180 miles",
                "Q: If today is Wednesday, what day was it 5 days ago? A: Counting back: Tuesday (1), Monday (2), Sunday (3), Saturday (4), Friday (5). It was Friday.",
                "Human: How do I improve my memory? Assistant: To improve memory: get adequate sleep, exercise regularly, practice spaced repetition, stay mentally active, and manage stress.",
                "Q: Explain why ice floats on water. A: Ice floats because it's less dense than liquid water. When water freezes, it expands and becomes lighter per unit volume.",
                "Q: What's 15% of 240? A: To find 15% of 240: 240 × 0.15 = 36",
            ],
            epochs=3,
            learning_rate=5e-5,
            batch_size=2,
            success_threshold=0.6,
            description="Multi-step reasoning and explanations"
        ),
        
        CurriculumStage(
            name="Expert Reasoning",
            difficulty=DifficultyLevel.EXPERT,
            data=[
                "Q: A compound interest problem: If you invest $1000 at 5% annual interest compounded yearly, how much will you have after 3 years? A: Using A = P(1+r)^t: A = 1000(1.05)³ = 1000 × 1.157625 = $1,157.63",
                "Human: Explain quantum entanglement simply. Assistant: Quantum entanglement is when two particles become connected so that measuring one instantly affects the other, regardless of distance. It's like having two magical coins that always land on opposite sides.",
                "Q: Why do we see lightning before hearing thunder? A: Light travels faster than sound. Lightning creates both light and sound simultaneously, but light reaches us almost instantly while sound travels at 343 m/s, creating the delay.",
                "Human: What's the best way to learn a new language? Assistant: Effective language learning combines: 1) Regular practice 2) Immersion when possible 3) Focus on practical vocabulary 4) Speaking from day one 5) Using multiple resources (apps, books, media) 6) Patience and consistency.",
                "Q: How does photosynthesis work? A: Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen. Chlorophyll in plant cells captures light energy, which drives chemical reactions that produce sugar for the plant and release oxygen as a byproduct.",
            ],
            epochs=4,
            learning_rate=3e-5,
            batch_size=3,
            success_threshold=0.5,
            description="Complex explanations and advanced reasoning"
        )
    ]


class AdaptiveCurriculumManager(CurriculumManager):
    """
    Adaptive curriculum manager that dynamically adjusts difficulty
    based on real-time performance metrics.
    """
    
    def __init__(self, stages: List[CurriculumStage], **kwargs):
        super().__init__(stages, **kwargs)
        self.performance_window = 10
        self.adaptation_rate = 0.1
        
    def adapt_stage_difficulty(self, performance_score: float):
        """Dynamically adapt current stage difficulty"""
        current_stage = self.current_stage
        
        if performance_score > current_stage.success_threshold + 0.1:
            # Performance is good, can increase difficulty slightly
            current_stage.learning_rate *= (1 + self.adaptation_rate)
            current_stage.success_threshold = min(0.9, current_stage.success_threshold + 0.05)
            
        elif performance_score < current_stage.success_threshold - 0.2:
            # Performance is poor, make it easier
            current_stage.learning_rate *= (1 - self.adaptation_rate)
            current_stage.success_threshold = max(0.3, current_stage.success_threshold - 0.05)
            
        logger.info(f"🔧 Adapted stage difficulty: LR={current_stage.learning_rate:.6f}, Threshold={current_stage.success_threshold:.3f}")


def create_domain_specific_curriculum(domain: str) -> List[CurriculumStage]:
    """Create curriculum for specific domains"""
    
    if domain == "math":
        return [
            CurriculumStage(
                name="Basic Arithmetic",
                difficulty=DifficultyLevel.BASIC,
                data=[
                    "Q: 3 + 4 = ? A: 7",
                    "Q: 10 - 6 = ? A: 4",
                    "Q: 5 × 3 = ? A: 15",
                    "Q: 12 ÷ 4 = ? A: 3",
                ],
                epochs=2,
                success_threshold=0.9
            ),
            CurriculumStage(
                name="Word Problems",
                difficulty=DifficultyLevel.INTERMEDIATE,
                data=[
                    "Q: Sarah has 12 apples. She gives 4 to her friend. How many does she have left? A: 12 - 4 = 8 apples",
                    "Q: A rectangle is 8 cm long and 5 cm wide. What's its area? A: Area = length × width = 8 × 5 = 40 cm²",
                ],
                epochs=3,
                success_threshold=0.7
            ),
            CurriculumStage(
                name="Advanced Problems",
                difficulty=DifficultyLevel.ADVANCED,
                data=[
                    "Q: Solve for x: 2x + 5 = 13 A: 2x = 13 - 5 = 8, so x = 4",
                    "Q: What's the derivative of x² + 3x? A: d/dx(x² + 3x) = 2x + 3",
                ],
                epochs=4,
                success_threshold=0.6
            )
        ]
    
    elif domain == "science":
        return [
            CurriculumStage(
                name="Basic Facts",
                difficulty=DifficultyLevel.BASIC,
                data=[
                    "Q: What gas do plants produce? A: Oxygen",
                    "Q: How many bones are in an adult human body? A: 206",
                    "Q: What's the chemical symbol for water? A: H₂O",
                ],
                epochs=2,
                success_threshold=0.8
            ),
            CurriculumStage(
                name="Scientific Reasoning",
                difficulty=DifficultyLevel.ADVANCED,
                data=[
                    "Q: Why do objects fall at the same rate in a vacuum? A: In a vacuum, there's no air resistance, so gravity accelerates all objects equally regardless of mass.",
                    "Q: How does DNA replication work? A: DNA unwinds, each strand serves as a template, and complementary nucleotides are added to form two identical double helixes.",
                ],
                epochs=4,
                success_threshold=0.6
            )
        ]
    
    else:
        return create_default_curriculum()
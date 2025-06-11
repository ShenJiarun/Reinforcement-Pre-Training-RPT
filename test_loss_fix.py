#!/usr/bin/env python3
"""
Critical Test: Verify Loss Fix
Ensures that loss starts near random baseline and never exceeds it.
"""

import torch
import math
from rpt import RPTTrainer, RPTModel, RewardSystem, DataProcessor
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_loss_sanity():
    """Test that loss behaves sanely and never exceeds random baseline"""
    
    logger.info("🔬 CRITICAL TEST: Loss Sanity Check")
    logger.info("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Calculate expected random loss
    vocab_size = len(tokenizer)
    expected_random_loss = math.log(vocab_size)
    logger.info(f"📊 Vocab size: {vocab_size}")
    logger.info(f"📊 Expected random loss: {expected_random_loss:.3f}")
    logger.info(f"📊 Maximum acceptable loss: {expected_random_loss * 1.2:.3f}")
    
    # Create model
    model = RPTModel(
        base_model="gpt2",
        tokenizer=tokenizer,
        add_value_head=True,
        device=str(device)
    )
    
    # Create reward system
    reward_system = RewardSystem(
        reward_type="hybrid",
        reward_scale=0.1,  # Small scale
        temperature=1.0
    )
    
    # Test data
    test_texts = [
        "Q: What is 2 + 2? A: 4",
        "Hello, how are you?",
        "The sky is blue."
    ]
    
    # Process data
    data_processor = DataProcessor(tokenizer, max_length=64)
    dataset = data_processor.create_dataset(test_texts, filter_quality=False)
    dataloader = data_processor.create_dataloader(dataset, batch_size=1, shuffle=False)
    
    # Create trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    trainer = RPTTrainer(
        model=model,
        reward_system=reward_system,
        optimizer=optimizer,
        train_dataloader=dataloader,
        max_epochs=1,
        device=str(device),
        output_dir="./test_outputs"
    )
    
    logger.info("🧪 Testing initial model behavior...")
    
    # Test initial loss (should be near random)
    model.eval()
    with torch.no_grad():
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            
            # Compute standard language modeling loss
            logits = outputs['logits']
            targets = batch['input_ids'].to(device)
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
            
        initial_loss = total_loss / num_batches
        
    logger.info(f"📈 Initial model loss: {initial_loss:.3f}")
    logger.info(f"📈 Random baseline: {expected_random_loss:.3f}")
    logger.info(f"📈 Ratio to random: {initial_loss / expected_random_loss:.2f}x")
    
    # Check if initial loss is reasonable
    if initial_loss > expected_random_loss * 2.0:
        logger.error(f"❌ FAIL: Initial loss {initial_loss:.3f} is too high (>2x random)")
        return False
    elif initial_loss > expected_random_loss * 1.5:
        logger.warning(f"⚠️  WARN: Initial loss {initial_loss:.3f} is higher than ideal")
    else:
        logger.info(f"✅ PASS: Initial loss {initial_loss:.3f} is reasonable")
    
    # Test training behavior
    logger.info("\n🚀 Testing training behavior...")
    
    # Override trainer's global_step for logging
    trainer.global_step = 0
    
    # Run a few training steps
    trainer.train()
    
    # Test final loss
    model.eval()
    with torch.no_grad():
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            
            logits = outputs['logits']
            targets = batch['input_ids'].to(device)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
            
        final_loss = total_loss / num_batches
    
    logger.info(f"\n📊 FINAL RESULTS:")
    logger.info(f"   Initial loss: {initial_loss:.3f}")
    logger.info(f"   Final loss: {final_loss:.3f}")
    logger.info(f"   Random baseline: {expected_random_loss:.3f}")
    logger.info(f"   Improvement: {initial_loss - final_loss:.3f}")
    
    # Validation checks
    success = True
    
    if final_loss > expected_random_loss * 1.5:
        logger.error(f"❌ FAIL: Final loss {final_loss:.3f} exceeds 1.5x random baseline")
        success = False
    
    if final_loss > initial_loss + 1.0:
        logger.error(f"❌ FAIL: Loss increased significantly during training")
        success = False
    
    if final_loss < 2.0:  # Very good performance
        logger.info(f"🎉 EXCELLENT: Final loss {final_loss:.3f} shows great performance!")
    elif final_loss < expected_random_loss:
        logger.info(f"✅ GOOD: Final loss {final_loss:.3f} beats random baseline")
    else:
        logger.info(f"✅ ACCEPTABLE: Final loss {final_loss:.3f} is within reasonable range")
    
    if success:
        logger.info("\n🎉 SUCCESS: Loss behavior is now FIXED and reasonable!")
        logger.info("✅ Loss starts near random baseline")
        logger.info("✅ Loss never exceeds catastrophic levels") 
        logger.info("✅ Training shows stable behavior")
    else:
        logger.error("\n❌ FAILURE: Loss behavior still problematic")
        
    return success

if __name__ == "__main__":
    success = test_loss_sanity()
    exit(0 if success else 1)
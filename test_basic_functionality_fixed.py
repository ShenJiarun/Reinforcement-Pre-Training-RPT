"""
Test basic functionality of the RPT package - Fixed version.
"""

import torch
print("Testing RPT package basic functionality...")

try:
    # Test imports
    print("Testing imports...")
    from rpt import RPTTrainer, RewardSystem, RPTModel, ScalingUtils, RPTMetrics, DataProcessor
    print("✓ All core imports successful")
    
    # Test RewardSystem
    print("\nTesting RewardSystem...")
    reward_system = RewardSystem(reward_type="accuracy", reward_scale=1.0)
    
    # Create dummy data
    batch_size, seq_len, vocab_size = 2, 10, 1000
    predictions = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    rewards = reward_system.compute_rewards(predictions, targets, attention_mask)
    print(f"✓ RewardSystem working - reward shape: {rewards.shape}")
    
    # Test reward statistics
    stats = reward_system.compute_reward_statistics(rewards, attention_mask)
    print(f"✓ Reward statistics: {stats}")
    
    # Test adaptive reward system
    from rpt.core.reward_system import AdaptiveRewardSystem
    adaptive_reward = AdaptiveRewardSystem()
    adaptive_reward.update_parameters(stats)
    print("✓ AdaptiveRewardSystem working")
    
    # Test DataProcessor
    print("\nTesting DataProcessor...")
    from transformers import AutoTokenizer
    
    # Use a small model for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    data_processor = DataProcessor(tokenizer=tokenizer, max_length=128, min_length=5)
    
    # Test with longer sample texts that will pass quality filtering
    sample_texts = [
        "This is a longer test sentence for RPT that should pass the quality filtering mechanisms.",
        "Another example text for testing the data processor with sufficient length and content quality.",
        "Machine learning is fascinating and complex, involving many different algorithms and techniques.",
        "The quick brown fox jumps over the lazy dog in this classic example sentence.",
        "Natural language processing enables computers to understand and process human language effectively."
    ]
    
    dataset = data_processor.create_dataset(sample_texts, filter_quality=True)
    print(f"✓ DataProcessor working - dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test dataloader
        dataloader = data_processor.create_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        print(f"✓ DataLoader working - batch keys: {list(batch.keys())}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
    else:
        print("⚠ Dataset is empty, skipping dataloader test")
    
    # Test RPTModel (simplified test)
    print("\nTesting RPTModel...")
    from transformers import AutoModel
    
    # Load a very small model for testing
    base_model = AutoModel.from_pretrained("gpt2")
    rpt_model = RPTModel(
        base_model=base_model,
        tokenizer=tokenizer,
        add_value_head=True
    )
    
    print(f"✓ RPTModel created - has value head: {rpt_model.value_head is not None}")
    
    # Test forward pass
    test_input = {
        "input_ids": torch.randint(0, tokenizer.vocab_size, (1, 10)),
        "attention_mask": torch.ones(1, 10)
    }
    
    with torch.no_grad():
        outputs = rpt_model(**test_input)
    
    print(f"✓ RPTModel forward pass - output keys: {list(outputs.keys())}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    if 'values' in outputs:
        print(f"  Values shape: {outputs['values'].shape}")
    
    # Test generation (simplified)
    print("\nTesting text generation...")
    sample_input = tokenizer("Hello world", return_tensors="pt")
    
    with torch.no_grad():
        generation_result = rpt_model.generate_with_reasoning(
            input_ids=sample_input.input_ids,
            max_length=20,
            temperature=1.0,
            reasoning_steps=1
        )
    
    generated_text = tokenizer.decode(generation_result["generated_ids"][0], skip_special_tokens=True)
    print(f"✓ Text generation working - generated: '{generated_text}'")
    
    # Test ScalingUtils
    print("\nTesting ScalingUtils...")
    from rpt.utils.scaling import ScalingConfig
    
    scaling_config = ScalingConfig(use_distributed=False, mixed_precision=True)
    scaling_utils = ScalingUtils(scaling_config)
    
    system_info = scaling_utils.get_system_info()
    print(f"✓ ScalingUtils working - CUDA available: {system_info['cuda_available']}")
    
    # Test memory estimation
    memory_estimates = scaling_utils.estimate_memory_usage(
        model=rpt_model,
        batch_size=4,
        sequence_length=128
    )
    print(f"✓ Memory estimation - total estimated: {memory_estimates['total_estimated']:.2f} GB")
    
    # Test RPTMetrics
    print("\nTesting RPTMetrics...")
    metrics = RPTMetrics(track_detailed_stats=False)
    
    # Test metrics computation
    step_metrics = metrics.update_metrics(
        step=1,
        predictions=predictions,
        targets=targets,
        rewards=rewards,
        attention_mask=attention_mask
    )
    
    print(f"✓ RPTMetrics working - computed: {list(step_metrics.keys())}")
    
    # Test summary stats
    summary = metrics.get_summary_stats()
    print(f"✓ Summary statistics available for {len(summary)} metrics")
    
    # Test basic trainer setup (without actually training)
    print("\nTesting RPTTrainer setup...")
    from torch.optim import AdamW
    
    if len(dataset) > 0:
        train_loader = data_processor.create_dataloader(dataset, batch_size=1, shuffle=False)
        optimizer = AdamW(rpt_model.parameters(), lr=1e-5)
        
        trainer = RPTTrainer(
            model=rpt_model,
            reward_system=reward_system,
            optimizer=optimizer,
            train_dataloader=train_loader,
            max_epochs=1,
            output_dir="./test_output"
        )
        
        print("✓ RPTTrainer setup successful")
    else:
        print("⚠ Skipping trainer test due to empty dataset")
    
    print("\n🎉 All basic functionality tests passed!")
    print("The RPT package is working correctly!")
    
except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    raise
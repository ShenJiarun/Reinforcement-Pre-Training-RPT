"""
Data Processing Utilities for RPT

This module provides utilities for processing and preparing text data
for Reinforcement Pre-Training, including tokenization, batching, and
data augmentation techniques.
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer
import json
import logging
from pathlib import Path
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class RPTDataset(Dataset):
    """
    Dataset class for RPT training that handles text tokenization
    and preparation for reinforcement learning.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        min_length: int = 10,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        reasoning_augmentation: bool = False
    ):
        """
        Initialize RPT dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            add_special_tokens: Whether to add special tokens
            return_attention_mask: Whether to return attention masks
            reasoning_augmentation: Whether to apply reasoning-based augmentation
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.add_special_tokens = add_special_tokens
        self.return_attention_mask = return_attention_mask
        self.reasoning_augmentation = reasoning_augmentation
        
        # Pre-tokenize texts for efficiency
        self.tokenized_texts = self._preprocess_texts()
        
    def _preprocess_texts(self) -> List[Dict[str, torch.Tensor]]:
        """Preprocess and tokenize all texts."""
        tokenized = []
        
        logger.info(f"Preprocessing {len(self.texts)} texts...")
        
        for i, text in enumerate(self.texts):
            try:
                # Tokenize text
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt"
                )
                
                # Check minimum length
                if encoding.input_ids.shape[1] >= self.min_length:
                    tokenized_item = {
                        "input_ids": encoding.input_ids.squeeze(0),
                        "text": text
                    }
                    
                    if self.return_attention_mask:
                        tokenized_item["attention_mask"] = encoding.attention_mask.squeeze(0)
                    
                    # Add reasoning augmentation if enabled
                    if self.reasoning_augmentation:
                        tokenized_item.update(self._apply_reasoning_augmentation(encoding, text))
                    
                    tokenized.append(tokenized_item)
                    
            except Exception as e:
                logger.warning(f"Error processing text {i}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(tokenized)} valid texts")
        return tokenized
    
    def _apply_reasoning_augmentation(
        self,
        encoding: Any,
        text: str
    ) -> Dict[str, torch.Tensor]:
        """
        Apply reasoning-based data augmentation.
        
        Args:
            encoding: Tokenized encoding
            text: Original text
            
        Returns:
            Augmentation features
        """
        input_ids = encoding.input_ids.squeeze(0)
        
        # Create reasoning targets by identifying important tokens
        # (This is a simplified approach; more sophisticated methods could be used)
        
        # Mark tokens that are likely important for reasoning
        reasoning_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Heuristic: mark tokens after certain keywords
        reasoning_keywords = ["because", "therefore", "however", "thus", "since", "due to"]
        
        for keyword in reasoning_keywords:
            keyword_tokens = self.tokenizer.encode(keyword, add_special_tokens=False)
            for i in range(len(input_ids) - len(keyword_tokens) + 1):
                if torch.equal(input_ids[i:i+len(keyword_tokens)], torch.tensor(keyword_tokens)):
                    # Mark next few tokens as reasoning-important
                    end_idx = min(i + len(keyword_tokens) + 3, len(input_ids))
                    reasoning_mask[i+len(keyword_tokens):end_idx] = True
        
        return {
            "reasoning_mask": reasoning_mask,
            "reasoning_weight": torch.where(reasoning_mask, 2.0, 1.0)  # Higher weight for reasoning tokens
        }
    
    def __len__(self) -> int:
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.tokenized_texts[idx]


class DataProcessor:
    """
    Main data processing utility for RPT training.
    """
    
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer],
        max_length: int = 512,
        min_length: int = 10,
        reasoning_augmentation: bool = False
    ):
        """
        Initialize data processor.
        
        Args:
            tokenizer: Tokenizer or model name
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            reasoning_augmentation: Whether to use reasoning augmentation
        """
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_length = max_length
        self.min_length = min_length
        self.reasoning_augmentation = reasoning_augmentation
        
    def load_text_data(
        self,
        data_path: Union[str, Path],
        data_format: str = "auto"
    ) -> List[str]:
        """
        Load text data from various formats.
        
        Args:
            data_path: Path to data file
            data_format: Data format ("txt", "json", "jsonl", "auto")
            
        Returns:
            List of text strings
        """
        data_path = Path(data_path)
        
        if data_format == "auto":
            if data_path.suffix == ".txt":
                data_format = "txt"
            elif data_path.suffix == ".json":
                data_format = "json"
            elif data_path.suffix == ".jsonl":
                data_format = "jsonl"
            else:
                raise ValueError(f"Cannot auto-detect format for {data_path}")
        
        logger.info(f"Loading data from {data_path} (format: {data_format})")
        
        if data_format == "txt":
            return self._load_txt(data_path)
        elif data_format == "json":
            return self._load_json(data_path)
        elif data_format == "jsonl":
            return self._load_jsonl(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _load_txt(self, path: Path) -> List[str]:
        """Load text from .txt file."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into paragraphs or sentences
        texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        return texts
    
    def _load_json(self, path: Path) -> List[str]:
        """Load text from .json file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List of strings or objects with text fields
            texts = []
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    texts.append(item['text'])
                elif isinstance(item, dict) and 'content' in item:
                    texts.append(item['content'])
        elif isinstance(data, dict) and 'texts' in data:
            texts = data['texts']
        else:
            raise ValueError("JSON format not recognized")
        
        return texts
    
    def _load_jsonl(self, path: Path) -> List[str]:
        """Load text from .jsonl file."""
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        if 'text' in item:
                            texts.append(item['text'])
                        elif 'content' in item:
                            texts.append(item['content'])
                except json.JSONDecodeError:
                    continue
        
        return texts
    
    def create_dataset(
        self,
        texts: List[str],
        split_ratio: Optional[float] = None,
        shuffle: bool = True,
        filter_quality: bool = True
    ) -> Union[RPTDataset, Tuple[RPTDataset, RPTDataset]]:
        """
        Create RPT dataset(s) from text data.
        
        Args:
            texts: List of text strings
            split_ratio: If provided, split into train/val (e.g., 0.9 for 90% train)
            shuffle: Whether to shuffle data
            filter_quality: Whether to apply quality filtering
            
        Returns:
            Dataset or tuple of (train_dataset, val_dataset)
        """
        # Filter quality if requested
        if filter_quality:
            texts = self._filter_text_quality(texts)
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(texts)
        
        # Split if requested
        if split_ratio is not None:
            split_idx = int(len(texts) * split_ratio)
            train_texts = texts[:split_idx]
            val_texts = texts[split_idx:]
            
            train_dataset = RPTDataset(
                texts=train_texts,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                min_length=self.min_length,
                reasoning_augmentation=self.reasoning_augmentation
            )
            
            val_dataset = RPTDataset(
                texts=val_texts,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                min_length=self.min_length,
                reasoning_augmentation=False  # No augmentation for validation
            )
            
            return train_dataset, val_dataset
        else:
            dataset = RPTDataset(
                texts=texts,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                min_length=self.min_length,
                reasoning_augmentation=self.reasoning_augmentation
            )
            
            return dataset
    
    def _filter_text_quality(self, texts: List[str]) -> List[str]:
        """
        Apply quality filtering to text data.
        
        Args:
            texts: Input texts
            
        Returns:
            Filtered texts
        """
        filtered_texts = []
        
        for text in texts:
            # Basic quality checks - made less strict
            if (
                len(text.strip()) >= max(self.min_length, 5) and  # At least 5 characters
                len(text.split()) >= 2 and  # At least 2 words (reduced from 3)
                not self._is_low_quality(text)
            ):
                filtered_texts.append(text.strip())
        
        logger.info(f"Filtered {len(texts)} -> {len(filtered_texts)} texts")
        
        # If no texts pass filtering, keep all original texts with a warning
        if len(filtered_texts) == 0:
            logger.warning("No texts passed quality filtering. Keeping all original texts.")
            return [text.strip() for text in texts if len(text.strip()) >= 1]
        
        return filtered_texts
    
    def _is_low_quality(self, text: str) -> bool:
        """Check if text is low quality."""
        # Simple heuristics for low-quality text - made less strict
        text_lower = text.lower()
        
        # Only reject if text is very short
        if len(text.strip()) < 3:
            return True
        
        # Check for too many repeated characters (increased threshold)
        if any(char * 6 in text for char in "abcdefghijklmnopqrstuvwxyz"):
            return True
        
        # Check for too many non-alphabetic characters (reduced threshold)
        if len(text) > 0:
            alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
            if alpha_ratio < 0.3:  # Reduced from 0.6 to 0.3
                return True
        
        # Check for spam-like patterns (only obvious spam)
        spam_indicators = ["click here now", "buy now!!!", "call now", "free money"]
        if any(indicator in text_lower for indicator in spam_indicators):
            return True
        
        return False
    
    def create_dataloader(
        self,
        dataset: RPTDataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        distributed: bool = False,
        **kwargs
    ) -> DataLoader:
        """
        Create DataLoader for RPT training.
        
        Args:
            dataset: RPT dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            distributed: Whether to use distributed sampling
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Configured DataLoader
        """
        # Check if dataset is empty
        if len(dataset) == 0:
            raise ValueError(
                "Dataset is empty. This usually means the text data was filtered out. "
                "Try using longer texts, reducing min_length, or disabling quality filtering."
            )
        
        # Collate function for padding
        def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            return self._collate_batch(batch)
        
        # Setup sampler for distributed training
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # Sampler handles shuffling
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )
    
    def _collate_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function to pad sequences in a batch.
        
        Args:
            batch: List of data items
            
        Returns:
            Padded batch
        """
        # Get all keys from the first item
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key == "text":
                # Keep text as list
                collated[key] = [item[key] for item in batch]
            elif key in ["input_ids", "attention_mask", "reasoning_mask", "reasoning_weight"]:
                # Pad sequences
                sequences = [item[key] for item in batch]
                
                # Determine padding value
                if key == "attention_mask":
                    pad_value = 0
                elif key == "reasoning_mask":
                    pad_value = False
                elif key == "reasoning_weight":
                    pad_value = 1.0
                else:  # input_ids
                    pad_value = self.tokenizer.pad_token_id
                
                # Pad sequences
                max_length = max(len(seq) for seq in sequences)
                padded_sequences = []
                
                for seq in sequences:
                    if len(seq) < max_length:
                        padding_length = max_length - len(seq)
                        if key == "reasoning_mask":
                            padding = torch.full((padding_length,), pad_value, dtype=torch.bool)
                        else:
                            padding = torch.full((padding_length,), pad_value, dtype=seq.dtype)
                        padded_seq = torch.cat([seq, padding])
                    else:
                        padded_seq = seq
                    
                    padded_sequences.append(padded_seq)
                
                collated[key] = torch.stack(padded_sequences)
            else:
                # For other types, just stack if possible
                try:
                    collated[key] = torch.stack([item[key] for item in batch])
                except:
                    collated[key] = [item[key] for item in batch]
        
        return collated
    
    def get_data_statistics(self, dataset: RPTDataset) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            dataset: RPT dataset
            
        Returns:
            Dataset statistics
        """
        lengths = []
        vocab_usage = defaultdict(int)
        
        for item in dataset:
            input_ids = item["input_ids"]
            lengths.append(len(input_ids))
            
            # Count vocabulary usage
            for token_id in input_ids.tolist():
                vocab_usage[token_id] += 1
        
        stats = {
            "total_samples": len(dataset),
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "length_percentiles": {
                "25th": np.percentile(lengths, 25),
                "50th": np.percentile(lengths, 50),
                "75th": np.percentile(lengths, 75),
                "90th": np.percentile(lengths, 90),
                "95th": np.percentile(lengths, 95)
            },
            "unique_tokens": len(vocab_usage),
            "total_tokens": sum(vocab_usage.values()),
            "vocab_coverage": len(vocab_usage) / self.tokenizer.vocab_size
        }
        
        return stats
    
    def debug_filtering(self, texts: List[str]) -> Dict[str, Any]:
        """
        Debug why texts might be filtered out.
        
        Args:
            texts: Input texts to analyze
            
        Returns:
            Debug information about filtering
        """
        debug_info = {
            "total_texts": len(texts),
            "too_short": 0,
            "too_few_words": 0,
            "low_quality": 0,
            "passed_filter": 0,
            "failed_examples": []
        }
        
        for i, text in enumerate(texts):
            text_stripped = text.strip()
            
            # Check each filter condition
            if len(text_stripped) < max(self.min_length, 5):
                debug_info["too_short"] += 1
                if len(debug_info["failed_examples"]) < 3:
                    debug_info["failed_examples"].append({
                        "text": text_stripped[:50] + "..." if len(text_stripped) > 50 else text_stripped,
                        "reason": f"too_short (length: {len(text_stripped)}, min: {max(self.min_length, 5)})"
                    })
            elif len(text_stripped.split()) < 2:
                debug_info["too_few_words"] += 1
                if len(debug_info["failed_examples"]) < 3:
                    debug_info["failed_examples"].append({
                        "text": text_stripped[:50] + "..." if len(text_stripped) > 50 else text_stripped,
                        "reason": f"too_few_words (words: {len(text_stripped.split())}, min: 2)"
                    })
            elif self._is_low_quality(text_stripped):
                debug_info["low_quality"] += 1
                if len(debug_info["failed_examples"]) < 3:
                    debug_info["failed_examples"].append({
                        "text": text_stripped[:50] + "..." if len(text_stripped) > 50 else text_stripped,
                        "reason": "low_quality"
                    })
            else:
                debug_info["passed_filter"] += 1
        
        return debug_info
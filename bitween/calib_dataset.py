import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any
import random


class CalibrationDataset:
    """
    A dataset class for loading and preprocessing calibration data for quantization.
    Supports various datasets including pile-10k with extensible architecture for more.
    """
    
    def __init__(
        self,
        dataset_name: str = "pile-10k",
        tokenizer=None,
        seqlen: int = 2048,
        nsamples: int = 512,
        seed: int = 42
    ):
        """
        Initialize calibration dataset.
        
        Args:
            dataset_name: Name of the dataset ('pile-10k', future: 'wikitext', 'c4', etc.)
            tokenizer: Tokenizer for text preprocessing
            seqlen: Sequence length for tokenization
            nsamples: Number of samples to extract
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seqlen = seqlen
        self.nsamples = nsamples
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.data = None
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and preprocess the specified dataset."""
        if self.dataset_name == "pile-10k":
            self._load_pile_10k()
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not supported yet. Available: 'pile-10k'")
    
    def _load_pile_10k(self):
        """Load and preprocess the pile-10k dataset."""
        print(f"Loading {self.dataset_name} dataset...")
        
        # Load the pile-10k dataset
        try:
            dataset = load_dataset("NeelNanda/pile-10k", split="train")
        except Exception as e:
            print(f"Error loading pile-10k: {e}")
            print("Falling back to mock data for testing...")
            self._create_mock_data()
            return
        
        # Extract text samples
        texts = []
        for i, sample in enumerate(dataset):
            if i >= self.nsamples:
                break
            texts.append(sample['text'])
        
        print(f"Loaded {len(texts)} text samples")
        
        if self.tokenizer is not None:
            self._tokenize_texts(texts)
        else:
            print("Warning: No tokenizer provided, storing raw texts")
            self.data = texts
    
    def _tokenize_texts(self, texts: List[str]):
        """Tokenize the text samples."""
        tokenized_samples = []
        
        for text in texts:
            # Tokenize the text
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.seqlen,
                truncation=True,
                padding=False
            )
            
            # Extract input_ids and ensure correct length
            input_ids = tokens["input_ids"].squeeze()
            
            if len(input_ids) >= self.seqlen:
                # Take first seqlen tokens
                input_ids = input_ids[:self.seqlen]
            else:
                # Pad if too short
                pad_length = self.seqlen - len(input_ids)
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
                ])
            
            tokenized_samples.append({
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids)
            })
        
        self.data = tokenized_samples
        print(f"Tokenized {len(tokenized_samples)} samples with sequence length {self.seqlen}")
    
    def _create_mock_data(self):
        """Create mock data for testing when real dataset is unavailable."""
        print("Creating mock calibration data...")
        
        if self.tokenizer is not None:
            vocab_size = len(self.tokenizer)
            mock_samples = []
            
            for _ in range(self.nsamples):
                input_ids = torch.randint(
                    0, min(vocab_size, 10000), (self.seqlen,), dtype=torch.long
                )
                mock_samples.append({
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids)
                })
            
            self.data = mock_samples
        else:
            # Create mock text data
            mock_texts = [
                f"This is mock calibration text sample number {i} for quantization testing."
                for i in range(self.nsamples)
            ]
            self.data = mock_texts
        
        print(f"Created {len(self.data)} mock samples")
    
    def get_dataloader(self, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        """
        Get a DataLoader for the calibration data.
        
        Args:
            batch_size: Batch size for the dataloader
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader instance
        """
        if self.data is None:
            raise ValueError("No data loaded. Call _load_dataset() first.")
        
        return DataLoader(
            self.data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn if self.tokenizer else None
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for tokenized data."""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """
        Get raw samples from the dataset.
        
        Args:
            num_samples: Number of samples to return (None for all)
            
        Returns:
            List of samples
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if num_samples is None:
            return self.data
        else:
            return self.data[:num_samples]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data) if self.data else 0
    
    def __getitem__(self, idx: int) -> Any:
        """Get a specific sample by index."""
        if self.data is None:
            raise ValueError("No data loaded.")
        return self.data[idx]


def get_calibration_dataset(
    dataset_name: str = "pile-10k",
    tokenizer=None,
    seqlen: int = 2048,
    nsamples: int = 512,
    seed: int = 42
) -> CalibrationDataset:
    """
    Convenience function to create a calibration dataset.
    
    Args:
        dataset_name: Name of the dataset
        tokenizer: Tokenizer for text preprocessing  
        seqlen: Sequence length for tokenization
        nsamples: Number of samples to extract
        seed: Random seed for reproducibility
        
    Returns:
        CalibrationDataset instance
    """
    return CalibrationDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seqlen=seqlen,
        nsamples=nsamples,
        seed=seed
    )
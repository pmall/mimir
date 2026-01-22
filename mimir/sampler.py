import math
import torch
from torch.utils.data import Sampler, Dataset


class LengthGroupedSampler(Sampler):
    """
    Sampler that groups sequences by length to minimize padding in batches.
    Strategy: Sort by length -> Chunk into batches -> Shuffle batch order.
    """

    def __init__(
        self, 
        dataset: Dataset, 
        batch_size: int, 
        drop_last: bool = False,
        generator=None
    ):
        """
        Args:
            dataset: Dataset (must implement get_lengths() or have len())
            batch_size: Size of batches
            drop_last: Whether to drop the last incomplete batch
            generator: Torch Generator for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator
        
        if not hasattr(dataset, "get_lengths"):
            raise ValueError("Dataset must implement get_lengths() for LengthGroupedSampler")

    def __iter__(self):
        # 1. Get lengths and sort indices
        lengths = self.dataset.get_lengths()
        
        # Shuffle indices first to ensuring random order within same-length groups
        indices = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        
        # Python's sort is stable
        indices.sort(key=lambda x: lengths[x])
        
        # 2. Chunk into batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                continue
            batches.append(batch)
            
        # 3. Shuffle the batches
        # We need to shuffle the list of lists
        # Use torch.randperm to select batch order
        batch_order = torch.randperm(len(batches), generator=self.generator).tolist()
        
        final_indices = []
        for i in batch_order:
            final_indices.extend(batches[i])
            
        return iter(final_indices)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size * self.batch_size
        else:
            return len(self.dataset)

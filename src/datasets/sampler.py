import torch
from torch.utils.data import Sampler


class BalancedClassSampler(Sampler):
    """Yield indices in exactly class-balanced consecutive mini-batches.

    The minority class is sampled with replacement. The sampler length equals
    the largest multiple of ``batch_size`` not exceeding the dataset size, so
    the number of optimizer steps per epoch stays comparable to ordinary
    shuffled training.
    """

    def __init__(self, labels, batch_size):
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError("Balanced sampling requires an even batch size >= 2")

        labels = torch.as_tensor(labels, dtype=torch.long)
        classes = torch.unique(labels)
        if classes.numel() != 2:
            raise ValueError("BalancedClassSampler requires exactly two classes")

        self.class_indices = [
            torch.nonzero(labels == class_index, as_tuple=False).flatten()
            for class_index in classes
        ]
        self.batch_size = batch_size
        self.samples_per_class = batch_size // 2
        self.n_batches = labels.numel() // batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            batch_parts = []
            for indices in self.class_indices:
                choices = torch.randint(
                    low=0,
                    high=indices.numel(),
                    size=(self.samples_per_class,),
                )
                batch_parts.append(indices[choices])
            batch = torch.cat(batch_parts)
            batch = batch[torch.randperm(self.batch_size)]
            yield from batch.tolist()

    def __len__(self):
        return self.n_batches * self.batch_size

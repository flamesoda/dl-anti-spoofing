import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    if not dataset_items:
        raise ValueError("Cannot collate an empty batch")

    result_batch = {}
    for key in dataset_items[0]:
        values = [item[key] for item in dataset_items]
        first = values[0]
        if isinstance(first, torch.Tensor):
            result_batch[key] = torch.stack(values)
        elif isinstance(first, (int, float, bool)):
            result_batch[key] = torch.tensor(values)
        else:
            # Keep metadata such as utterance and attack identifiers on CPU.
            result_batch[key] = values
    return result_batch

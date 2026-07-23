import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Spectrograms have identical shapes thanks to FixedLengthCrop, so they
    are stacked along a new batch dimension. Raw waveforms have variable
    lengths and are not needed by the model, so they are not collated.
    Utterance and attack identifiers are passed through as lists of
    strings for the submission writer and per-attack metrics.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["spectrogram"] = torch.stack(
        [elem["spectrogram"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])
    result_batch["utt_id"] = [elem["utt_id"] for elem in dataset_items]
    result_batch["attack_id"] = [elem["attack_id"] for elem in dataset_items]

    return result_batch

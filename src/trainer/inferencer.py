import csv

import torch
from tqdm.auto import tqdm

from src.metrics.eer import EERMetric
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model and compute the bonafide score.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker | None): unused, kept for signature
                compatibility.
            part (str): name of the partition.
        Returns:
            batch (dict): dict-based batch with model outputs and scores.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)
        batch["scores"] = batch["logits"].softmax(dim=-1)[:, 1]

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a partition, save per-utterance scores as a CSV for
        grading, and compute the EER using the partition labels.

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """
        self.is_train = False
        self.model.eval()

        utt_ids = []
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=None,
                )
                utt_ids.extend(batch["utt_id"])
                all_scores.append(batch["scores"].cpu())
                all_labels.append(batch["labels"].cpu())

        all_scores = torch.cat(all_scores).numpy()
        all_labels = torch.cat(all_labels).numpy()

        self._save_predictions(part, utt_ids, all_scores)

        return {"EER": EERMetric()(scores=all_scores, labels=all_labels)}

    def _save_predictions(self, part, utt_ids, scores):
        """
        Write per-utterance scores as a two-column, header-less CSV
        (utterance id, score), the format expected by grading.py.

        Args:
            part (str): name of the partition, used as the file name.
            utt_ids (list[str]): utterance identifiers.
            scores (np.ndarray): bonafide scores aligned with utt_ids.
        """
        save_file = self.save_path / f"{part}.csv"
        with open(save_file, "w", newline="") as f:
            writer = csv.writer(f)
            for utt_id, score in zip(utt_ids, scores):
                writer.writerow([utt_id, float(score)])

from pathlib import Path

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

LABEL_MAP = {"bonafide": 1, "spoof": 0}

PARTITION_DIRS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}

PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


class ASVSpoofDataset(BaseDataset):
    """
    Dataset for the LA partition of ASVspoof2019.
    """

    def __init__(self, part, data_dir, *args, **kwargs):
        """
        Args:
            part (str): "train", "dev", "eval".
            data_dir (str): path to the root of the LA partition on disk
        """
        assert part in PARTITION_DIRS, f"Unknown partition: {part}"

        index_dir = ROOT_PATH / "data" / "asvspoof" / part
        index_path = index_dir / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(part, data_dir)
            index_dir.mkdir(exist_ok=True, parents=True)
            write_json(index, str(index_path))

        super().__init__(index, *args, **kwargs)

    def _create_index(self, part, data_dir):
        """
        Parse the CM protocol file of the given partition and build the index.

        Each protocol line has the format:
            SPEAKER_ID  UTT_ID  -  ATTACK_ID  LABEL

        Args:
            part (str): "train", "dev", "eval".
            data_dir (str): path to the root of the LA partition on disk.
        Returns:
            index (list[dict])
        """
        data_dir = Path(data_dir)
        protocol_path = (
            data_dir / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_FILES[part]
        )
        audio_dir = data_dir / PARTITION_DIRS[part] / "flac"

        index = []
        with open(protocol_path) as f:
            for line in f:
                speaker_id, utt_id, _, attack_id, label = line.strip().split()
                index.append(
                    {
                        "path": str(audio_dir / f"{utt_id}.flac"),
                        "label": LABEL_MAP[label],
                        "utt_id": utt_id,
                        "attack_id": attack_id,
                    }
                )
        return index

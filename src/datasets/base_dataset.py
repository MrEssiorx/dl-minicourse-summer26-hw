import logging
import random
from typing import List

import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for audio datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.

    Each dataset element is a dict that carries the loaded waveform, its
    spectrogram, the target label, and all extra metadata fields found in
    the index entry (e.g. utterance and attack identifiers).
    """

    def __init__(
        self,
        index,
        target_sr=16000,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
        cache=False,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            target_sr (int): supported sample rate. Loaded audio is
                resampled to target_sr if needed.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the tensor name.
                The special key "get_spectrogram" defines the
                waveform-to-spectrogram front-end.
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

        self._cache = {} if cache else None

    def __getitem__(self, ind):
        """
        Get element from the index, load its audio, compute the spectrogram,
        and combine everything into a dict.

        All metadata fields of the index entry are passed through into the
        instance dict, so downstream consumers (collate_fn, logger,
        submission writer) can access them by name.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        if self._cache is not None and ind in self._cache:
            return self._cache[ind]

        data_dict = self._index[ind]

        instance_data = dict(data_dict)
        instance_data["audio"] = self.load_audio(data_dict["path"])
        instance_data["labels"] = data_dict["label"]
        instance_data = self.preprocess_data(instance_data)

        if self._cache is not None:
            self._cache[ind] = instance_data

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        """
        Load audio from disk and keep the first channel only.

        The sample rate is asserted to match target_sr because the STFT
        front-end parameters are defined in milliseconds at target_sr.

        Args:
            path (str): path to the audio file.
        Returns:
            audio (Tensor): waveform of shape (1, T), T varies between
                utterances.
        """
        audio_tensor, sr = torchaudio.load(path)
        assert sr == self.target_sr, f"Expected {self.target_sr} Hz, got {sr}: {path}"
        return audio_tensor[0:1, :]

    def get_spectrogram(self, audio):
        """
        Special instance transform with a special key to
        get spectrogram from audio.

        Args:
            audio (Tensor): original audio.
        Returns:
            spectrogram (Tensor): spectrogram for the audio.
        """
        return self.instance_transforms["get_spectrogram"](audio)

    def preprocess_data(self, instance_data):
        """
        Compute the spectrogram from the waveform and apply
        spectrogram-level transforms.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) with the spectrogram added.
        """
        if self.instance_transforms is None:
            return instance_data

        instance_data["spectrogram"] = self.get_spectrogram(instance_data["audio"])

        if "spectrogram" in self.instance_transforms:
            instance_data["spectrogram"] = self.instance_transforms["spectrogram"](
                instance_data["spectrogram"]
            )

        return instance_data

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "label" in entry, (
                "Each dataset item should include field 'label'"
                " - object ground-truth label."
            )

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index

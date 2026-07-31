from pathlib import Path

import torch
import torchaudio

from src.datasets.base_dataset import BaseDataset

LABEL_TO_INDEX = {"spoof": 0, "bonafide": 1}


class ASVspoof2019LADataset(BaseDataset):
    """Logical Access partition of the ASVspoof 2019 corpus.

    Only files listed in the official CM protocol are indexed. This matters
    because some distributed corpus archives contain extra audio files that
    are not evaluation trials.
    """

    PROTOCOL_FILENAMES = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    def __init__(
        self,
        root_dir,
        partition,
        feature_extractor,
        protocol_path=None,
        audio_dir=None,
        expected_sample_rate=16000,
        balanced_limit=None,
        balanced_sampling=False,
        *args,
        **kwargs,
    ):
        if partition not in self.PROTOCOL_FILENAMES:
            raise ValueError("partition must be one of: train, dev, eval")

        self.root_dir = Path(root_dir).expanduser()
        self.partition = partition
        self.feature_extractor = feature_extractor
        self.expected_sample_rate = expected_sample_rate
        self.balanced_sampling = balanced_sampling

        if protocol_path is None:
            protocol_path = (
                self.root_dir
                / "ASVspoof2019_LA_cm_protocols"
                / self.PROTOCOL_FILENAMES[partition]
            )
        if audio_dir is None:
            audio_dir = self.root_dir / f"ASVspoof2019_LA_{partition}" / "flac"

        self.protocol_path = Path(protocol_path).expanduser()
        self.audio_dir = Path(audio_dir).expanduser()
        index = self._create_index()
        if balanced_limit is not None:
            index = self._make_balanced_subset(index, balanced_limit)
        super().__init__(index=index, *args, **kwargs)

    @staticmethod
    def _make_balanced_subset(index, total_size):
        if total_size < 2 or total_size % 2 != 0:
            raise ValueError("balanced_limit must be an even integer >= 2")
        per_class = total_size // 2
        bona = [item for item in index if item["label"] == LABEL_TO_INDEX["bonafide"]]
        spoof = [item for item in index if item["label"] == LABEL_TO_INDEX["spoof"]]
        if len(bona) < per_class or len(spoof) < per_class:
            raise ValueError("Not enough examples to construct a balanced subset")
        balanced = []
        for bona_item, spoof_item in zip(bona[:per_class], spoof[:per_class]):
            balanced.extend((bona_item, spoof_item))
        return balanced

    def _create_index(self):
        if not self.protocol_path.is_file():
            raise FileNotFoundError(f"Protocol not found: {self.protocol_path}")
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        index = []
        with self.protocol_path.open("r", encoding="utf-8") as protocol:
            for line_number, line in enumerate(protocol, start=1):
                fields = line.strip().split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(
                        f"Malformed protocol line {line_number} in "
                        f"{self.protocol_path}: {line.rstrip()}"
                    )
                speaker_id, utterance_id, _, attack_id, label_name = fields
                if label_name not in LABEL_TO_INDEX:
                    raise ValueError(
                        f"Unknown label '{label_name}' at line {line_number}"
                    )
                audio_path = self.audio_dir / f"{utterance_id}.flac"
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Audio listed in protocol is missing: {audio_path}"
                    )
                index.append(
                    {
                        "path": str(audio_path),
                        "label": LABEL_TO_INDEX[label_name],
                        "label_name": label_name,
                        "speaker_id": speaker_id,
                        "utterance_id": utterance_id,
                        "attack_id": attack_id,
                    }
                )
        if not index:
            raise ValueError(f"Protocol is empty: {self.protocol_path}")
        return index

    @property
    def labels(self):
        """Integer labels in protocol order, used by the train sampler."""
        return [item["label"] for item in self._index]

    def __getitem__(self, index):
        item = self._index[index]
        waveform, sample_rate = torchaudio.load(item["path"])
        if sample_rate != self.expected_sample_rate:
            raise ValueError(
                f"{item['path']} has sample rate {sample_rate}; expected "
                f"{self.expected_sample_rate}"
            )

        spectrogram = self.feature_extractor(waveform)
        instance = {
            "spectrogram": spectrogram,
            "labels": item["label"],
            "utterance_id": item["utterance_id"],
            "speaker_id": item["speaker_id"],
            "attack_id": item["attack_id"],
        }
        return self.preprocess_data(instance)

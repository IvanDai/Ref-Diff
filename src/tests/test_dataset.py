from pathlib import Path

import h5py
import yaml

from rfstyle.data.dataset import PairedSignalDataset
from rid2026 import RID2026DatasetGenerator


def test_generate_and_filter(tmp_path: Path):
    config = yaml.safe_load(Path("configs/generate_rid2026_smoke.yaml").read_text())
    config["signal"]["frame_length"] = 128
    config["signal"]["modulations"] = ["BPSK"]
    config["impairments"]["esn0_db"] = [0]
    config["splits"] = {"train": 1, "valid": 1, "test": 1}
    RID2026DatasetGenerator(config).generate(tmp_path)
    dataset = PairedSignalDataset(tmp_path, "train", filters={"profiles": ["awgn_only"]})
    item = dataset[0]
    assert item["condition"].shape == (2, 128)
    assert item["target"].shape == (2, 128)
    assert item["esn0_db"] == 0.0
    with h5py.File(tmp_path / "shards" / dataset.rows[0]["shard"], "r") as handle:
        assert {"x_pure", "x_tx", "y_rx", "channel_taps"} <= set(handle)


def test_full_rid2026_configuration_matches_rml2018_scale():
    config = yaml.safe_load(
        Path("configs/generate_rid2026_full.yaml").read_text()
    )
    assert RID2026DatasetGenerator(config).example_count() == 2_555_904

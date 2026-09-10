from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np

from .generator import RFSignalGenerator, SignalConfig, derive_seed
from .impairments import sample_parameters
from .waveforms import RML2018_MODULATIONS


def complex_to_iq(x: np.ndarray) -> np.ndarray:
    return np.stack((x.real, x.imag), axis=0).astype(np.float32)


class RID2026DatasetGenerator:
    """Stream RID2026 paired records into HDF5 shards and a CSV manifest."""

    def __init__(self, config: dict):
        self.config = config
        self.frame_length = int(config["signal"]["frame_length"])
        self.max_taps = int(config["impairments"]["max_channel_taps"])
        self.seed = int(config.get("seed", 233))
        self.shard_size = int(config["shard_size"])
        if self.shard_size < 1:
            raise ValueError("shard_size must be positive")

    def example_count(self) -> int:
        """Return the configured total without materializing sample records."""
        modulation_count = len(
            self.config["signal"].get("modulations") or RML2018_MODULATIONS
        )
        esn0_count = len(self.config["impairments"]["esn0_db"])
        total_conditions = 0
        for counts in self.config["splits"].values():
            if isinstance(counts, dict):
                total_conditions += sum(int(value) for value in counts.values())
            else:
                total_conditions += int(counts) * len(self.config["profiles"])
        return modulation_count * esn0_count * total_conditions

    def generate(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        shard_dir = output / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (output / "dataset.json").write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        fields = [
            "sample_id", "split", "shard", "row", "profile", "modulation",
            "esn0_db", "samples_per_symbol", "rrc_alpha", "timing_offset",
            "symbol_rate_offset", "phase_offset", "carrier_offset",
            "delay_spread", "gmsk_bt", "am_modulation_index", "fm_deviation",
            "sample_seed",
        ]
        with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            sample_id = 0
            for split_index, (split, counts) in enumerate(self.config["splits"].items()):
                batch: list[tuple[str, str, float, int]] = []
                shard_index = 0
                for record in self._iter_records(counts, split_index):
                    batch.append(record)
                    if len(batch) == self.shard_size:
                        sample_id = self._commit_shard(
                            shard_dir, writer, split, shard_index, batch, sample_id
                        )
                        shard_index += 1
                        batch = []
                if batch:
                    sample_id = self._commit_shard(
                        shard_dir, writer, split, shard_index, batch, sample_id
                    )

    def _iter_records(
        self, counts: int | dict, split_index: int
    ) -> Iterator[tuple[str, str, float, int]]:
        signal_cfg = self.config["signal"]
        mods = signal_cfg.get("modulations") or list(RML2018_MODULATIONS)
        esn0_values = self.config["impairments"]["esn0_db"]
        for profile_index, profile in enumerate(self.config["profiles"]):
            per_condition = int(
                counts.get(profile, 0) if isinstance(counts, dict) else counts
            )
            for mod_index, modulation in enumerate(mods):
                for esn0_index, esn0_db in enumerate(esn0_values):
                    for repeat in range(per_condition):
                        seed = derive_seed(
                            self.seed,
                            split_index,
                            profile_index,
                            mod_index,
                            esn0_index,
                            repeat,
                        )
                        yield profile, modulation, float(esn0_db), seed

    def _commit_shard(
        self,
        shard_dir: Path,
        writer: csv.DictWriter,
        split: str,
        shard_index: int,
        batch: list[tuple[str, str, float, int]],
        sample_id: int,
    ) -> int:
        shard_name = f"{split}-{shard_index:05d}.h5"
        rows = self._write_shard(shard_dir / shard_name, batch)
        for row, metadata in enumerate(rows):
            metadata.update(
                sample_id=sample_id, split=split, shard=shard_name, row=row
            )
            writer.writerow(metadata)
            sample_id += 1
        return sample_id

    def _write_shard(
        self, path: Path, records: list[tuple[str, str, float, int]]
    ) -> list[dict]:
        count = len(records)
        pure = np.empty((count, 2, self.frame_length), dtype=np.float32)
        transmitted = np.empty_like(pure)
        received = np.empty_like(pure)
        channels = np.empty((count, self.max_taps, 2), dtype=np.float32)
        metadata = []
        signal_cfg = self.config["signal"]
        impairment_cfg = self.config["impairments"]

        for row, (profile, modulation, esn0_db, seed) in enumerate(records):
            waveform_config_rng = np.random.default_rng(derive_seed(seed, 0))
            impairment_rng = np.random.default_rng(derive_seed(seed, 1))
            waveform_seed = derive_seed(seed, 2)
            sps = int(waveform_config_rng.choice(signal_cfg["samples_per_symbol"]))
            alpha_low, alpha_high = signal_cfg["rrc_alpha"]
            alpha = float(waveform_config_rng.uniform(alpha_low, alpha_high))
            generator = RFSignalGenerator(
                SignalConfig(
                    frame_length=self.frame_length,
                    samples_per_symbol=sps,
                    guard_symbols=int(signal_cfg.get("guard_symbols", 32)),
                    rrc_alpha=alpha,
                    gmsk_bt=float(signal_cfg.get("gmsk_bt", 0.3)),
                    message_bandwidth=float(signal_cfg.get("message_bandwidth", 0.2)),
                    am_modulation_index=float(signal_cfg.get("am_modulation_index", 1.0)),
                    fm_deviation=float(signal_cfg.get("fm_deviation", 0.35)),
                )
            )
            sample_impairments = dict(impairment_cfg)
            sample_impairments["enabled"] = dict(impairment_cfg.get("enabled", {}))
            for key, value in self.config["profiles"][profile].items():
                if key == "enabled":
                    sample_impairments["enabled"].update(value)
                else:
                    sample_impairments[key] = value
            sample_impairments["esn0_db"] = [esn0_db]
            params = sample_parameters(sample_impairments, self.max_taps, impairment_rng)
            result = generator.generate(modulation, params, seed=waveform_seed)

            pure[row] = complex_to_iq(result.x_pure)
            transmitted[row] = complex_to_iq(result.parameters.x_tx)
            received[row] = complex_to_iq(result.y_rx)
            channels[row] = complex_to_iq(params.channel_taps).T
            waveform = result.parameters.waveform
            metadata.append({
                "profile": profile,
                "modulation": modulation,
                "esn0_db": params.esn0_db,
                "samples_per_symbol": waveform.samples_per_symbol,
                "rrc_alpha": waveform.rrc_alpha,
                "gmsk_bt": waveform.gmsk_bt,
                "am_modulation_index": waveform.am_modulation_index,
                "fm_deviation": waveform.fm_deviation,
                "timing_offset": params.timing_offset,
                "symbol_rate_offset": params.symbol_rate_offset,
                "phase_offset": params.phase_offset,
                "carrier_offset": params.carrier_offset,
                "delay_spread": params.delay_spread,
                "sample_seed": seed,
            })

        temporary = path.with_suffix(path.suffix + ".tmp")
        with h5py.File(temporary, "w") as handle:
            chunk = (min(count, 64), 2, self.frame_length)
            handle.create_dataset("x_pure", data=pure, chunks=chunk, compression="gzip")
            handle.create_dataset("x_tx", data=transmitted, chunks=chunk, compression="gzip")
            handle.create_dataset("y_rx", data=received, chunks=chunk, compression="gzip")
            handle.create_dataset("channel_taps", data=channels, compression="gzip")
        temporary.replace(path)
        return metadata


def generate_dataset(config: dict, output_dir: str | Path) -> None:
    RID2026DatasetGenerator(config).generate(output_dir)

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections.abc import Callable, Iterator
from pathlib import Path

import h5py
import numpy as np

from .generator import RFSignalGenerator, SignalConfig, derive_seed
from .impairments import sample_parameters
from .waveforms import RML2018_MODULATIONS


def complex_to_iq(x: np.ndarray) -> np.ndarray:
    return np.stack((x.real, x.imag), axis=-2).astype(np.float32)


_WORKER_GENERATOR: RID2026DatasetGenerator | None = None


def _initialize_worker(config: dict) -> None:
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = RID2026DatasetGenerator(config)


def _generate_worker_batch(task: tuple) -> tuple[tuple[int, int, int, int], dict]:
    if _WORKER_GENERATOR is None:
        raise RuntimeError("RID2026 worker was not initialized")
    mod_index, modulation, esn0_index, esn0_db, start, stop = task
    batch = _WORKER_GENERATOR._generate_batch(
        mod_index, modulation, esn0_index, esn0_db, range(start, stop)
    )
    return (mod_index, esn0_index, start, stop), batch


class RID2026DatasetGenerator:
    """Stream RID2026 records into one structured HDF5 file."""

    def __init__(self, config: dict):
        self.config = config
        self.frame_length = int(config["signal"]["frame_length"])
        self.max_taps = int(config["impairments"]["max_channel_taps"])
        self.seed = int(config.get("seed", 233))
        self.write_batch_size = int(config.get("write_batch_size", 64))
        if self.write_batch_size < 1:
            raise ValueError("write_batch_size must be positive")
        self.examples_per_condition = int(config["examples_per_condition"])
        if self.examples_per_condition < 1:
            raise ValueError("examples_per_condition must be positive")
        self.workers = int(config.get("workers", 1))
        if self.workers < 1:
            raise ValueError("workers must be positive")
        self.compression = config.get("compression")
        if self.compression not in {None, "gzip", "lzf"}:
            raise ValueError("compression must be null, gzip, or lzf")
        self.modulations = tuple(
            config["signal"].get("modulations") or RML2018_MODULATIONS
        )
        self.esn0_values = tuple(float(x) for x in config["impairments"]["esn0_db"])

    def example_count(self) -> int:
        return (
            len(self.modulations)
            * len(self.esn0_values)
            * self.examples_per_condition
        )

    def generate(
        self,
        output_file: str | Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        output = Path(output_file)
        if output.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("output_file must end with .h5 or .hdf5")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".incomplete")

        with h5py.File(temporary, "w") as handle:
            datasets = self._create_datasets(handle)
            completed = 0
            for location, batch in self._iter_batches():
                mod_index, esn0_index, start, stop = location
                selection = (mod_index, esn0_index, slice(start, stop))
                for name, values in batch.items():
                    datasets[name][selection] = values
                completed += stop - start
                if progress is not None:
                    progress(completed, self.example_count())

        temporary.replace(output)

    def _tasks(self) -> Iterator[tuple]:
        for mod_index, modulation in enumerate(self.modulations):
            for esn0_index, esn0_db in enumerate(self.esn0_values):
                for start in range(
                    0, self.examples_per_condition, self.write_batch_size
                ):
                    stop = min(
                        start + self.write_batch_size, self.examples_per_condition
                    )
                    yield (
                        mod_index, modulation, esn0_index, esn0_db, start, stop
                    )

    def _iter_batches(self) -> Iterator[tuple[tuple[int, int, int, int], dict]]:
        if self.workers == 1:
            for task in self._tasks():
                mod_index, modulation, esn0_index, esn0_db, start, stop = task
                yield (mod_index, esn0_index, start, stop), self._generate_batch(
                    mod_index,
                    modulation,
                    esn0_index,
                    esn0_db,
                    range(start, stop),
                )
            return

        tasks = iter(self._tasks())
        with ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_initialize_worker,
            initargs=(self.config,),
        ) as executor:
            pending = set()
            for _ in range(self.workers * 2):
                task = next(tasks, None)
                if task is None:
                    break
                pending.add(executor.submit(_generate_worker_batch, task))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(executor.submit(_generate_worker_batch, task))

    def _create_datasets(self, handle: h5py.File) -> dict[str, h5py.Dataset]:
        modulation_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "modulation",
            data=np.asarray(self.modulations, dtype=object),
            dtype=modulation_dtype,
        )
        handle.create_dataset(
            "esn0_db", data=np.asarray(self.esn0_values, dtype=np.float32)
        )

        leading = (
            len(self.modulations),
            len(self.esn0_values),
            self.examples_per_condition,
        )
        batch_chunk = min(self.write_batch_size, self.examples_per_condition)
        signal_chunks = (1, 1, batch_chunk, 2, self.frame_length)
        storage_options = {"compression": self.compression}
        if self.compression is not None:
            storage_options["shuffle"] = True
        datasets = {
            "y_rx": handle.create_dataset(
                "y_rx",
                shape=leading + (2, self.frame_length),
                dtype="f4",
                chunks=signal_chunks,
                **storage_options,
            ),
            "x_ref": handle.create_dataset(
                "x_ref",
                shape=leading + (2, self.frame_length),
                dtype="f4",
                chunks=signal_chunks,
                **storage_options,
            ),
        }

        parameters = handle.create_group("parameters")
        scalar_chunks = (1, 1, batch_chunk)
        for name, dtype in (
            ("samples_per_symbol", "u2"),
            ("rrc_alpha", "f4"),
            ("timing_offset", "f4"),
            ("symbol_rate_offset", "f4"),
            ("phase_offset", "f4"),
            ("carrier_offset", "f4"),
            ("delay_spread", "f4"),
            ("channel_path_count", "u2"),
        ):
            datasets[name] = parameters.create_dataset(
                name,
                shape=leading,
                dtype=dtype,
                chunks=scalar_chunks,
                **storage_options,
            )
        datasets["channel_taps"] = parameters.create_dataset(
            "channel_taps",
            shape=leading + (self.max_taps, 2),
            dtype="f4",
            chunks=(1, 1, batch_chunk, self.max_taps, 2),
            **storage_options,
        )
        return datasets

    def _generate_batch(
        self,
        mod_index: int,
        modulation: str,
        esn0_index: int,
        esn0_db: float,
        example_indices: range,
    ) -> dict[str, np.ndarray]:
        count = len(example_indices)
        batch = {
            "y_rx": np.empty((count, 2, self.frame_length), dtype=np.float32),
            "x_ref": np.empty((count, 2, self.frame_length), dtype=np.float32),
            "samples_per_symbol": np.empty(count, dtype=np.uint16),
            "rrc_alpha": np.empty(count, dtype=np.float32),
            "timing_offset": np.empty(count, dtype=np.float32),
            "symbol_rate_offset": np.empty(count, dtype=np.float32),
            "phase_offset": np.empty(count, dtype=np.float32),
            "carrier_offset": np.empty(count, dtype=np.float32),
            "delay_spread": np.empty(count, dtype=np.float32),
            "channel_path_count": np.empty(count, dtype=np.uint16),
            "channel_taps": np.empty(
                (count, self.max_taps, 2), dtype=np.float32
            ),
        }
        signal_cfg = self.config["signal"]
        impairment_cfg = self.config["impairments"]

        for row, example_index in enumerate(example_indices):
            seed = derive_seed(
                self.seed,
                mod_index,
                esn0_index,
                example_index,
            )
            waveform_config_rng = np.random.default_rng(derive_seed(seed, 0))
            impairment_rng = np.random.default_rng(derive_seed(seed, 1))
            waveform_seed = derive_seed(seed, 2)
            samples_per_symbol = int(
                waveform_config_rng.choice(signal_cfg["samples_per_symbol"])
            )
            alpha_low, alpha_high = signal_cfg["rrc_alpha"]
            rrc_alpha = float(waveform_config_rng.uniform(alpha_low, alpha_high))
            generator = RFSignalGenerator(
                SignalConfig(
                    frame_length=self.frame_length,
                    samples_per_symbol=samples_per_symbol,
                    guard_symbols=int(signal_cfg.get("guard_symbols", 32)),
                    rrc_alpha=rrc_alpha,
                    gmsk_bt=float(signal_cfg.get("gmsk_bt", 0.3)),
                    message_bandwidth=float(signal_cfg.get("message_bandwidth", 0.2)),
                    am_modulation_index=float(
                        signal_cfg.get("am_modulation_index", 1.0)
                    ),
                    fm_deviation=float(signal_cfg.get("fm_deviation", 0.35)),
                )
            )
            sample_impairments = dict(impairment_cfg)
            sample_impairments["enabled"] = dict(
                impairment_cfg.get("enabled", {})
            )
            sample_impairments["esn0_db"] = [esn0_db]
            impairments = sample_parameters(
                sample_impairments, self.max_taps, impairment_rng
            )
            result = generator.generate(modulation, impairments, seed=waveform_seed)
            waveform = result.parameters.waveform

            batch["y_rx"][row] = complex_to_iq(result.y_rx)
            batch["x_ref"][row] = complex_to_iq(result.x_ref)
            batch["samples_per_symbol"][row] = waveform.samples_per_symbol
            batch["rrc_alpha"][row] = (
                np.nan if waveform.rrc_alpha is None else waveform.rrc_alpha
            )
            batch["timing_offset"][row] = impairments.timing_offset
            batch["symbol_rate_offset"][row] = impairments.symbol_rate_offset
            batch["phase_offset"][row] = impairments.phase_offset
            batch["carrier_offset"][row] = impairments.carrier_offset
            batch["delay_spread"][row] = impairments.delay_spread
            batch["channel_path_count"][row] = impairments.channel_path_count
            batch["channel_taps"][row] = complex_to_iq(
                impairments.channel_taps
            ).T

        return batch


def generate_dataset(config: dict, output_file: str | Path) -> None:
    RID2026DatasetGenerator(config).generate(output_file)

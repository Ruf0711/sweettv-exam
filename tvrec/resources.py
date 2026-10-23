"""Observable progress and measured process resource use."""

import logging
import threading
import time
from contextlib import contextmanager

import psutil

LOGGER = logging.getLogger("tvrec")


class Resources:
    def __init__(self):
        self.process = psutil.Process()
        self.started = time.perf_counter()
        self.sampled_peak_rss = 0
        self.samples = []
        self.phases = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def __enter__(self):
        self.sample("start")
        self._thread.start()
        return self

    def _sample_loop(self):
        while not self._stop.wait(0.1):
            self.sampled_peak_rss = max(self.sampled_peak_rss, self.process.memory_info().rss)

    def sample(self, label):
        rss = self.process.memory_info().rss
        self.sampled_peak_rss = max(self.sampled_peak_rss, rss)
        self.samples.append({"phase": label, "elapsed_seconds": time.perf_counter() - self.started, "rss_bytes": rss})

    @contextmanager
    def phase(self, name):
        started = time.perf_counter()
        LOGGER.info("Starting %s", name)
        yield
        self.phases[name] = time.perf_counter() - started
        self.sample(name)
        LOGGER.info("Completed %s in %.2fs; RSS %.1f MiB", name, self.phases[name], self.samples[-1]["rss_bytes"] / 2**20)

    def report(self):
        memory = self.process.memory_info()
        return {
            "elapsed_seconds": time.perf_counter() - self.started,
            "phase_seconds": dict(self.phases),
            "sampled_peak_rss_bytes": max(self.sampled_peak_rss, memory.rss),
            "sampling_interval_seconds": 0.1,
            "os_peak_working_set_bytes": getattr(memory, "peak_wset", None),
            "os_peak_note": "Windows lifetime process peak, including imports" if hasattr(memory, "peak_wset") else "Unavailable on this platform; sampled RSS is reported",
            "physical_ram_bytes": psutil.virtual_memory().total,
            "phase_memory_samples": list(self.samples),
        }

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()


def progress(label, rows, total=None):
    LOGGER.info("%s: %s%s rows", label, f"{rows:,}", " / " + f"{total:,}" if total is not None else "")

"""速度測定用のプロファイラー"""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import psutil
import pynvml
import torch


class SpeedProfiler:
    """学習速度を測定してCSVに記録するクラス"""

    _instance: ClassVar["SpeedProfiler | None"] = None
    _initialized: ClassVar[bool] = False

    def __init__(self):
        if SpeedProfiler._initialized:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = Path(f"speed_profile_{timestamp}.csv")
        self.usage_path = Path(f"speed_profile_{timestamp}_usage.csv")
        self._start_time = time.time()
        self._stop_monitoring = False

        with self.output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["elapsed_time", "event", "epoch", "iteration"])

        with self.usage_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["elapsed_time", "cpu_percent", "gpu_percent", "gpu_memory_mb"]
            )

        self._epoch = 0
        self._iteration = 0

        self._nvml_initialized = False
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

        self._monitoring_thread = threading.Thread(
            target=self._monitor_usage, daemon=True
        )
        self._monitoring_thread.start()

        SpeedProfiler._initialized = True

    @classmethod
    def get_instance(cls) -> "SpeedProfiler":
        """シングルトンインスタンスを取得"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_epoch(self, epoch: int) -> None:
        """エポック番号を設定"""
        self._epoch = epoch

    def set_iteration(self, iteration: int) -> None:
        """イテレーション番号を設定"""
        self._iteration = iteration

    def record(self, event: str) -> None:
        """イベントを記録"""
        elapsed_time = time.time() - self._start_time
        with self.output_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([elapsed_time, event, self._epoch, self._iteration])

    def _monitor_usage(self) -> None:
        """CPU/GPU使用率を0.05秒ごとに記録"""
        process = psutil.Process()

        while not self._stop_monitoring:
            elapsed_time = time.time() - self._start_time

            cpu_percent = process.cpu_percent(interval=None)

            gpu_percent = 0.0
            gpu_memory_mb = 0.0

            if self._nvml_initialized:
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    gpu_percent = float(utilization.gpu)

                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    gpu_memory_mb = memory_info.used / (1024 * 1024)
                except Exception:
                    pass
            elif torch.cuda.is_available():
                try:
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                except Exception:
                    pass

            with self.usage_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([elapsed_time, cpu_percent, gpu_percent, gpu_memory_mb])

            time.sleep(0.025)

    def stop(self) -> None:
        """モニタリングを停止"""
        self._stop_monitoring = True


def get_profiler() -> SpeedProfiler:
    """グローバルプロファイラーを取得"""
    return SpeedProfiler.get_instance()

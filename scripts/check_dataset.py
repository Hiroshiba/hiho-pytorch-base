"""データセットをチェックする"""

import argparse
import os

from torch.utils.data import RandomSampler
from tqdm import tqdm
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data_loader import create_prepared_data_loader
from hiho_pytorch_base.dataset import Dataset, create_dataset


def check_dataset(config_yaml_path: UPath, trials: int) -> None:
    """データセットの整合性をチェックする"""
    config = Config.load(config_yaml_path)

    num_workers = config.train.preprocess_workers
    if num_workers is None:
        num_workers = os.cpu_count()
        if num_workers is None:
            raise ValueError("Failed to get CPU count")

    datasets = create_dataset(config.dataset)
    try:
        for i in range(trials):
            print(f"try {i}")
            _check(
                datasets.train,
                desc="train",
                config=config,
                drop_last=True,
                num_workers=num_workers,
            )
            _check(
                datasets.test,
                desc="test",
                config=config,
                drop_last=False,
                num_workers=num_workers,
            )
            if datasets.eval is not None:
                _check(
                    datasets.eval,
                    desc="eval",
                    config=config,
                    drop_last=False,
                    num_workers=num_workers,
                )
            if datasets.valid is not None:
                _check(
                    datasets.valid,
                    desc="valid",
                    config=config,
                    drop_last=False,
                    num_workers=num_workers,
                )
    finally:
        datasets.close()


def _check(
    dataset: Dataset,
    desc: str,
    config: Config,
    drop_last: bool,
    num_workers: int,
) -> None:
    batch_size = config.train.batch_size
    data_loader = create_prepared_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=RandomSampler(dataset),
        num_workers=num_workers,
        pin_memory=config.train.use_gpu,
        drop_last=drop_last,
    )
    for _, _ in tqdm(
        enumerate(data_loader), desc=desc, total=len(dataset) // batch_size
    ):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=UPath)
    parser.add_argument("--trials", type=int, default=3)
    check_dataset(**vars(parser.parse_args()))

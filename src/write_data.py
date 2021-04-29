import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, List

DATA_PATH = 'runs'


@dataclass
class ImageStats:
    filename: str
    iterations: Optional[int] = 0
    mean: Optional[List[float]] = field(default_factory=list)
    std: Optional[List[float]] = field(default_factory=list)
    ssim: Optional[List[float]] = field(default_factory=list)
    ratio: Optional[List[float]] = field(default_factory=list)

    def append_iteration(self, mean: float, std: float, ssim: float, ratio: float):
        self.mean.append(mean)
        self.std.append(std)
        self.ssim.append(ssim)
        self.ratio.append(ratio)
        self.iterations += 1


@dataclass
class RunStats:
    algorithm: str
    date: Optional[datetime.date] = field(default_factory=datetime.datetime.utcnow)
    images: Optional[List[ImageStats]] = field(default_factory=list)

    def append_image_stats(self, image_stats: ImageStats) -> None:
        self.images.append(image_stats)


def default(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    return obj


def dump_json(obj: Any, file_path: str) -> None:
    with open(file_path, mode='w+', encoding='utf-8') as data_file:
        json.dump(obj, data_file, default=default)


def get_latest_file_path(algorithm: str) -> str:
    return f'{DATA_PATH}/{algorithm}.json'


def get_run_file_path(run: RunStats) -> str:
    return f'{DATA_PATH}/{run.algorithm}/{run.date.isoformat().replace(".", "-").replace(":", "-")}.json'


def write_to_current(run_data: RunStats) -> None:
    file_path = get_latest_file_path(run_data.algorithm)
    dump_json(run_data, file_path)


def write_to_past(run_data: RunStats) -> None:
    algorithm_name = run_data.algorithm
    Path(f'{DATA_PATH}/{algorithm_name}').mkdir(parents=True, exist_ok=True)
    file_path = get_run_file_path(run_data)
    dump_json(run_data, file_path)


def write_data(run_data: RunStats) -> None:
    write_to_current(run_data)
    write_to_past(run_data)

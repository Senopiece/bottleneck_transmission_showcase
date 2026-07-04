from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass(frozen=True)
class VideoItem:
    path: Path
    kind: str
    name: str


def repo_root_from_tools() -> Path:
    return Path(__file__).resolve().parents[3]


def default_dataset_dir() -> Path:
    return repo_root_from_tools() / "receiver" / "datasets" / "raw" / "videos"


def default_derived_dir() -> Path:
    return repo_root_from_tools() / "receiver" / "datasets" / "derived"


def discover_videos(dataset_dir: Path) -> list[VideoItem]:
    items: list[VideoItem] = []
    for kind in ("good", "bad"):
        folder = dataset_dir / kind
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                items.append(VideoItem(path=path, kind=kind, name=path.stem))
    return items

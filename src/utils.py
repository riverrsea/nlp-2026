from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

FONT_5X7 = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    "-": [0, 0, 0, 31, 0, 0, 0],
    "0": [14, 17, 19, 21, 25, 17, 14],
    "1": [4, 12, 4, 4, 4, 4, 14],
    "2": [14, 17, 1, 2, 4, 8, 31],
    "3": [30, 1, 1, 14, 1, 1, 30],
    "4": [2, 6, 10, 18, 31, 2, 2],
    "5": [31, 16, 16, 30, 1, 1, 30],
    "6": [14, 16, 16, 30, 17, 17, 14],
    "7": [31, 1, 2, 4, 8, 8, 8],
    "8": [14, 17, 17, 14, 17, 17, 14],
    "9": [14, 17, 17, 15, 1, 1, 14],
    "A": [14, 17, 17, 31, 17, 17, 17],
    "B": [30, 17, 17, 30, 17, 17, 30],
    "C": [14, 17, 16, 16, 16, 17, 14],
    "D": [30, 17, 17, 17, 17, 17, 30],
    "E": [31, 16, 16, 30, 16, 16, 31],
    "F": [31, 16, 16, 30, 16, 16, 16],
    "G": [14, 17, 16, 16, 19, 17, 14],
    "H": [17, 17, 17, 31, 17, 17, 17],
    "I": [14, 4, 4, 4, 4, 4, 14],
    "J": [7, 2, 2, 2, 2, 18, 12],
    "K": [17, 18, 20, 24, 20, 18, 17],
    "L": [16, 16, 16, 16, 16, 16, 31],
    "M": [17, 27, 21, 21, 17, 17, 17],
    "N": [17, 25, 21, 19, 17, 17, 17],
    "O": [14, 17, 17, 17, 17, 17, 14],
    "P": [30, 17, 17, 30, 16, 16, 16],
    "Q": [14, 17, 17, 17, 21, 18, 13],
    "R": [30, 17, 17, 30, 20, 18, 17],
    "S": [15, 16, 16, 14, 1, 1, 30],
    "T": [31, 4, 4, 4, 4, 4, 4],
    "U": [17, 17, 17, 17, 17, 17, 14],
    "V": [17, 17, 17, 17, 17, 10, 4],
    "W": [17, 17, 17, 21, 21, 21, 10],
    "X": [17, 17, 10, 4, 10, 17, 17],
    "Y": [17, 17, 10, 4, 4, 4, 4],
    "Z": [31, 1, 2, 4, 8, 16, 31],
}


def setup_logging(level: str = "INFO", name: str = "data_prepare") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger

    logger.setLevel(level.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def create_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_directories(paths: Iterable[str | Path]) -> None:
    for path in paths:
        create_dir(path)


def ensure_file_exists(path: str | Path, description: str = "File") -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    return file_path


def set_seed(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def extract_label_from_dirname(dirname: str) -> str:
    return dirname.split("-", maxsplit=1)[0].strip()


def find_text_files(data_dir: str | Path) -> List[Path]:
    base_dir = Path(data_dir)
    files = [
        path
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".txt"
    ]
    return sorted(files, key=lambda item: item.as_posix())


def read_text_file(
    file_path: str | Path,
    encodings: Sequence[str],
    logger: logging.Logger | None = None,
) -> str | None:
    path = Path(file_path)
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            if logger is not None:
                logger.warning("Failed to read %s: %s", path, exc)
            return None

    if logger is not None:
        logger.warning(
            "Failed to decode %s with encodings: %s",
            path,
            ", ".join(encodings),
        )
    return None


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = CONTROL_CHAR_PATTERN.sub("", text)

    cleaned_lines: List[str] = []
    previous_blank = False
    for raw_line in text.split("\n"):
        normalized_line = re.sub(r"[ \u3000]+", " ", raw_line).strip()
        if normalized_line:
            cleaned_lines.append(normalized_line)
            previous_blank = False
        else:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True

    cleaned_text = "\n".join(cleaned_lines).strip()
    return cleaned_text


def write_csv(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    create_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_json(data: Mapping[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    create_dir(path.parent)

    def _json_default(value: object) -> str:
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )


def read_csv_rows(input_path: str | Path) -> list[dict[str, str]]:
    path = ensure_file_exists(input_path, description="CSV file")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def save_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    predictions: Sequence[int],
    id2label: Mapping[int, str],
    output_path: str | Path,
) -> None:
    if len(rows) != len(predictions):
        raise ValueError("The number of rows and predictions must match.")

    prediction_rows: list[dict[str, object]] = []
    for row, pred_label_id in zip(rows, predictions):
        prediction_rows.append(
            {
                "text": row.get("text", ""),
                "label": row.get("label", ""),
                "label_id": row.get("label_id", ""),
                "pred_label": id2label[int(pred_label_id)],
                "pred_label_id": int(pred_label_id),
                "file_path": row.get("file_path", ""),
            }
        )

    write_csv(
        prediction_rows,
        fieldnames=[
            "text",
            "label",
            "label_id",
            "pred_label",
            "pred_label_id",
            "file_path",
        ],
        output_path=output_path,
    )


def summarize_rows_by_label(
    rows: Sequence[Mapping[str, object]],
    label_key: str = "label",
) -> dict[str, int]:
    counts = Counter(str(row[label_key]) for row in rows)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def resolve_device(requested_device: str = "auto") -> str:
    normalized = requested_device.lower()
    if normalized != "auto":
        return requested_device

    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def save_training_curve(
    history: Mapping[str, Sequence[float]],
    output_path: str | Path,
    title: str = "Training Curve",
) -> None:
    if not history:
        return

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    path = Path(output_path)
    create_dir(path.parent)
    plt.figure(figsize=(10, 6))
    for metric_name, values in history.items():
        if not values:
            continue
        epochs = list(range(1, len(values) + 1))
        plt.plot(epochs, list(values), marker="o", label=metric_name)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_metric_curve(
    values: Sequence[float],
    output_path: str | Path,
    title: str,
    y_label: str,
) -> None:
    if not values:
        return

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    path = Path(output_path)
    create_dir(path.parent)
    epochs = list(range(1, len(values) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, list(values), marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_confusion_matrix_figure(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    if not matrix or not labels:
        return

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        import seaborn as sns  # type: ignore
    except Exception:
        return

    path = Path(output_path)
    create_dir(path.parent)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        np.asarray(matrix, dtype=int),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(labels),
        yticklabels=list(labels),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def quantile(values: Sequence[int], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of an empty sequence.")
    if not 0.0 <= q <= 1.0:
        raise ValueError("Quantile must be in the range [0, 1].")

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return float(lower_value)
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def summarize_numeric_series(values: Sequence[int]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence.")

    ordered = sorted(values)
    count = len(ordered)
    mean_value = sum(ordered) / count
    midpoint = count // 2
    if count % 2 == 0:
        median_value = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    else:
        median_value = float(ordered[midpoint])

    return {
        "min": int(ordered[0]),
        "max": int(ordered[-1]),
        "mean": round(mean_value, 4),
        "median": round(median_value, 4),
        "q50": round(quantile(ordered, 0.50), 4),
        "q75": round(quantile(ordered, 0.75), 4),
        "q90": round(quantile(ordered, 0.90), 4),
        "q95": round(quantile(ordered, 0.95), 4),
    }


class SimpleCanvas:
    def __init__(
        self,
        width: int,
        height: int,
        background: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset : offset + 3] = bytes(color)

    def fill_rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        left = max(0, min(x0, x1))
        right = min(self.width, max(x0, x1))
        top = max(0, min(y0, y1))
        bottom = min(self.height, max(y0, y1))
        for y in range(top, bottom):
            row_start = (y * self.width + left) * 3
            row_end = (y * self.width + right) * 3
            self.pixels[row_start:row_end] = bytes(color) * (right - left)

    def draw_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy

        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            double_error = 2 * error
            if double_error >= dy:
                error += dy
                x0 += sx
            if double_error <= dx:
                error += dx
                y0 += sy

    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int] = (0, 0, 0),
        scale: int = 2,
    ) -> None:
        cursor_x = x
        for char in text.upper():
            pattern = FONT_5X7.get(char, FONT_5X7[" "])
            for row_index, row_pattern in enumerate(pattern):
                for column_index in range(5):
                    if row_pattern & (1 << (4 - column_index)):
                        self.fill_rect(
                            cursor_x + column_index * scale,
                            y + row_index * scale,
                            cursor_x + (column_index + 1) * scale,
                            y + (row_index + 1) * scale,
                            color,
                        )
            cursor_x += 6 * scale

    def save_png(self, output_path: str | Path) -> None:
        path = Path(output_path)
        create_dir(path.parent)

        raw_rows = []
        row_size = self.width * 3
        for row_index in range(self.height):
            start = row_index * row_size
            end = start + row_size
            raw_rows.append(b"\x00" + bytes(self.pixels[start:end]))
        raw_data = b"".join(raw_rows)
        compressed = zlib.compress(raw_data, level=9)

        def png_chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png_signature = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)

        with path.open("wb") as file:
            file.write(png_signature)
            file.write(png_chunk(b"IHDR", ihdr))
            file.write(png_chunk(b"IDAT", compressed))
            file.write(png_chunk(b"IEND", b""))


def _draw_axes(
    canvas: SimpleCanvas,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int] = (60, 66, 80),
) -> None:
    canvas.draw_line(left, top, left, bottom, color)
    canvas.draw_line(left, bottom, right, bottom, color)


def _split_label(label: str, max_line_length: int = 8) -> List[str]:
    if len(label) <= max_line_length:
        return [label]
    midpoint = math.ceil(len(label) / 2)
    return [label[:midpoint], label[midpoint:]]


def save_class_distribution_chart(
    class_counts: Mapping[str, int],
    output_path: str | Path,
    width: int = 1400,
    height: int = 900,
) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.font_manager import FontProperties  # type: ignore
    except Exception:
        return

    path = Path(output_path)
    create_dir(path.parent)
    font_path = Path("font") / "SIMFANG.TTF"
    font_prop = FontProperties(fname=str(font_path.resolve())) if font_path.exists() else None

    label_to_zh = {
        "art": "艺术",
        "computer": "计算机",
        "economy": "经济",
        "transportation": "交通",
        "education": "教育",
        "environment": "环境",
        "sports": "体育",
        "military": "军事",
        "politics": "政治",
        "medicine": "医药",
    }
    labels = list(class_counts.keys())
    display_labels = [label_to_zh.get(label, label) for label in labels]
    values = [int(class_counts[label]) for label in labels]
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111)
    bars = ax.bar(display_labels, values, color="#4a87d7", edgecolor="#2e5fa3")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=15,
            fontproperties=font_prop,
        )

    if font_prop is not None:
        ax.set_title("类别分布", fontsize=25, pad=18, fontproperties=font_prop)
        ax.set_xlabel("种类", fontsize=20, labelpad=16, fontproperties=font_prop)
        ax.set_ylabel("样本数量", fontsize=20, rotation=90, fontproperties=font_prop)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(font_prop)
    else:
        ax.set_title("类别分布", fontsize=25, pad=18, fontproperties=font_prop)
        ax.set_xlabel("种类", fontsize=20, labelpad=16, fontproperties=font_prop)
        ax.set_ylabel("样本数量", fontsize=20, rotation=90, fontproperties=font_prop)

    # Keep the y-axis title centered on the y-axis and move it slightly right
    # so it is visible inside the figure area.
    ax.yaxis.set_label_coords(-0.03, 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.1, right=0.98)
    fig.savefig(path)
    plt.close(fig)


def save_length_distribution_chart(
    lengths: Sequence[int],
    output_path: str | Path,
    width: int = 1400,
    height: int = 900,
    bins: int = 30,
) -> None:
    if not lengths:
        raise ValueError("Lengths cannot be empty when drawing a histogram.")
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    path = Path(output_path)
    create_dir(path.parent)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.hist(list(lengths), bins=max(5, int(bins)), color="#4a87d7", edgecolor="#2e5fa3", alpha=0.9)
    ax.set_title("Text Length Distribution", fontsize=18, pad=18)
    ax.set_xlabel("Character Count", fontsize=12, labelpad=16)
    ax.set_ylabel("Frequency", fontsize=12, rotation=90)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.1, right=0.98)
    fig.savefig(path)
    plt.close(fig)

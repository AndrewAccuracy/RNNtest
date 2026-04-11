from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required to run this experiment.\n"
        "Install it with: pip install -r requirements.txt"
    ) from exc


OUTPUT_DIR = Path("outputs")
# 用于 Phase2（RNN vs LSTM）拉开差距：max gap 建议 ≥ 18
RECOMMENDED_GAP_MAX_FOR_PHASE2 = 18
PRESET_SLIDES_GAPS = (5, 10, 18, 20, 22, 26, 30, 50, 100)
PRESET_SLIDES_DENSE_GAPS = tuple(range(5, 31, 2)) + (35, 40, 50, 70, 100)
PRESET_SLIDES_SEEDS = (1, 2, 3, 4, 5)
PRESET_QUICK_GAPS = (5, 10)
PRESET_QUICK_SEEDS = (1,)

# 热力图纵轴顺序（上→下）；Flat-MLP 固定在最底行，便于和递归模型对比
HEATMAP_ROW_ORDER = ("History-MLP", "RNN", "LSTM", "Flat-MLP")

VOCAB = ["A", "B", "c", "d", "e", "f", "g", "?"]
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
DISTRACTOR_TOKENS = ("c", "d", "e", "f", "g")


def reseed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Config:
    gap_lengths: tuple[int, ...] = (5, 10, 18, 20, 22, 26, 30, 50, 100)
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    train_samples_per_gap: int = 2000
    test_samples_per_gap: int = 500
    batch_size: int = 32
    epochs: int = 30
    training_loss_epoch_start: int = 12
    learning_rate: float = 1e-3
    hidden_size: int = 16
    history_window: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Phase 2: multi-token delayed memory
    phase2_prefix_len: int = 2
    acc_threshold_critical_gap: float = 0.85
    train_loss_threshold: float = 0.1
    match_lstm_params: bool = False
    phase2_data_dir: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Historical model vs RNN vs LSTM on a delayed-memory task")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--training-loss-epoch-start",
        type=int,
        default=12,
        metavar="E",
        help="Training-loss figures only plot from this epoch onward (1-based); hides early flat region.",
    )
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gaps", type=int, nargs="+", default=[5, 10, 18, 20, 22, 26, 30, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--export-data",
        action="store_true",
        help="Write train/test JSONL under --data-export-root before training (same RNG as first seed + 100/200).",
    )
    parser.add_argument(
        "--data-export-root",
        type=Path,
        default=Path("data/delayed_memory"),
        help="Root directory for --export-data (gap_XXX/train.jsonl, test.jsonl).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="If set, load fixed train/test JSONL from gap_XXX/ for all seeds (reproducible archive).",
    )
    parser.add_argument(
        "--preset",
        choices=("none", "slides", "slides-dense", "quick"),
        default="none",
        help="slides: full gap list + 5 seeds. slides-dense: finer gaps + 5 seeds. quick: gaps 5,10 seed 1.",
    )
    parser.add_argument(
        "--phase2-prefix-len",
        type=int,
        default=2,
        metavar="K",
        help="Phase 2: first K tokens are A/B pattern; num_classes=2^K (keep small, e.g. 2–3).",
    )
    parser.add_argument(
        "--acc-threshold-critical-gap",
        type=float,
        default=0.85,
        metavar="A",
        help="Phase 2: critical gap = smallest gap with mean test accuracy below this.",
    )
    parser.add_argument(
        "--train-loss-threshold",
        type=float,
        default=0.1,
        metavar="L",
        help="Phase 2: first epoch where train loss < L counts as time-to-threshold (null if never).",
    )
    parser.add_argument(
        "--match-lstm-params",
        action="store_true",
        help="Phase 2: choose LSTM hidden size so total params are closest to RNN (same K and num_classes).",
    )
    parser.add_argument(
        "--phase2-data-dir",
        type=Path,
        default=None,
        help="Fixed Phase2 JSONL root: phase2_kK/gap_XXX/train.jsonl. Default: same as --data-dir if set.",
    )
    return parser.parse_args()


def apply_cli_preset(args: argparse.Namespace) -> None:
    if args.preset == "slides":
        args.gaps = list(PRESET_SLIDES_GAPS)
        args.seeds = list(PRESET_SLIDES_SEEDS)
    elif args.preset == "slides-dense":
        args.gaps = list(PRESET_SLIDES_DENSE_GAPS)
        args.seeds = list(PRESET_SLIDES_SEEDS)
    elif args.preset == "quick":
        args.gaps = list(PRESET_QUICK_GAPS)
        args.seeds = list(PRESET_QUICK_SEEDS)


def one_hot(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(token_ids, num_classes=vocab_size).float()


class DelayedMemoryDataset(Dataset):
    def __init__(self, gap_length: int, sample_count: int, seed: int) -> None:
        rng = random.Random(seed)
        samples = []
        labels = []

        for _ in range(sample_count):
            first_token = rng.choice(["A", "B"])
            distractors = [rng.choice(DISTRACTOR_TOKENS) for _ in range(gap_length)]
            sequence = [first_token, *distractors, "?"]
            label = 0 if first_token == "A" else 1

            samples.append([TOKEN_TO_ID[token] for token in sequence])
            labels.append(label)

        self.sequences = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class MultiTokenDelayedMemoryDataset(Dataset):
    """前 K 个 token 为 A/B 模式，中间 gap 个干扰，末尾 ?；标签为模式类别 0..2^K-1（位权 2^i，i=0 为序列首 token）。"""

    def __init__(self, prefix_len: int, gap_length: int, sample_count: int, seed: int) -> None:
        if prefix_len < 1:
            raise ValueError("prefix_len must be >= 1")
        rng = random.Random(seed)
        num_classes = 2**prefix_len
        samples = []
        labels = []

        for _ in range(sample_count):
            pattern = [rng.choice(["A", "B"]) for _ in range(prefix_len)]
            label = 0
            for i, tok in enumerate(pattern):
                if tok == "B":
                    label |= 1 << i
            distractors = [rng.choice(DISTRACTOR_TOKENS) for _ in range(gap_length)]
            sequence = [*pattern, *distractors, "?"]
            if len(sequence) != prefix_len + gap_length + 1:
                raise RuntimeError("sequence length mismatch")

            samples.append([TOKEN_TO_ID[token] for token in sequence])
            labels.append(label)

        self.sequences = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.prefix_len = prefix_len
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def id_sequence_to_tokens(token_ids: list[int]) -> list[str]:
    id_to_tok = {idx: t for t, idx in TOKEN_TO_ID.items()}
    return [id_to_tok[i] for i in token_ids]


class DelayedMemoryDatasetFromFiles(Dataset):
    """从 export 生成的 JSONL 加载；Phase2 行可含 prefix_len，若传入 expected_prefix_len 则校验。"""

    def __init__(self, jsonl_path: Path, expected_prefix_len: int | None = None) -> None:
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {jsonl_path}")

        samples: list[list[int]] = []
        labels: list[int] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "token_ids" in row:
                    samples.append([int(x) for x in row["token_ids"]])
                elif "tokens" in row:
                    samples.append([TOKEN_TO_ID[t] for t in row["tokens"]])
                else:
                    raise ValueError(f"JSONL row must have token_ids or tokens: {jsonl_path}")
                labels.append(int(row["label"]))
                if expected_prefix_len is not None and "prefix_len" in row:
                    if int(row["prefix_len"]) != expected_prefix_len:
                        raise ValueError(
                            f"prefix_len mismatch in {jsonl_path}: row has {row['prefix_len']}, "
                            f"expected {expected_prefix_len}"
                        )

        self.sequences = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def export_delayed_memory_jsonl(config: Config, data_root: Path) -> None:
    """按 gap 写入 train/test；RNG 与内存模式、且 seed=config.seeds[0] 时一致（train: seed0+100, test: seed0+200）。"""
    base_seed = config.seeds[0]
    data_root.mkdir(parents=True, exist_ok=True)

    for gap_length in config.gap_lengths:
        gap_dir = data_root / f"gap_{gap_length:03d}"
        gap_dir.mkdir(parents=True, exist_ok=True)

        train_ds = DelayedMemoryDataset(gap_length, config.train_samples_per_gap, base_seed + 100)
        test_ds = DelayedMemoryDataset(gap_length, config.test_samples_per_gap, base_seed + 200)

        _write_jsonl_dataset(gap_dir / "train.jsonl", train_ds)
        _write_jsonl_dataset(gap_dir / "test.jsonl", test_ds)


def export_phase2_multi_token_jsonl(config: Config, data_root: Path) -> None:
    """写入 data_root/phase2_kK/gap_XXX/train|test.jsonl，RNG 与内存模式一致。"""
    base_seed = config.seeds[0]
    k = config.phase2_prefix_len
    sub_root = data_root / f"phase2_k{k}"
    sub_root.mkdir(parents=True, exist_ok=True)

    for gap_length in config.gap_lengths:
        gap_dir = sub_root / f"gap_{gap_length:03d}"
        gap_dir.mkdir(parents=True, exist_ok=True)

        train_ds = MultiTokenDelayedMemoryDataset(
            k, gap_length, config.train_samples_per_gap, base_seed + 100
        )
        test_ds = MultiTokenDelayedMemoryDataset(
            k, gap_length, config.test_samples_per_gap, base_seed + 200
        )

        _write_jsonl_phase2_dataset(gap_dir / "train.jsonl", train_ds)
        _write_jsonl_phase2_dataset(gap_dir / "test.jsonl", test_ds)


def _write_jsonl_dataset(path: Path, dataset: DelayedMemoryDataset) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            seq, lab = dataset[i]
            token_ids = seq.tolist()
            row = {
                "token_ids": token_ids,
                "tokens": id_sequence_to_tokens(token_ids),
                "label": int(lab.item()),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl_phase2_dataset(path: Path, dataset: MultiTokenDelayedMemoryDataset) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            seq, lab = dataset[i]
            token_ids = seq.tolist()
            row = {
                "token_ids": token_ids,
                "tokens": id_sequence_to_tokens(token_ids),
                "label": int(lab.item()),
                "prefix_len": dataset.prefix_len,
                "num_classes": dataset.num_classes,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class FlatSequenceMLP(nn.Module):
    """全长序列 one-hot 展平后过 MLP（无时间递归）。同一 gap 下序列长度固定为 gap_length+2。"""

    def __init__(self, vocab_size: int, seq_len: int, hidden_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(vocab_size * seq_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence = one_hot(token_ids, len(VOCAB))
        features = sequence.reshape(token_ids.size(0), -1)
        return self.net(features)


class FixedWindowMLP(nn.Module):
    """固定窗口「历史」基线：只取序列末尾 history_window 个 token 的 one-hot，拼成向量后过 MLP。

    对应演示里的 History-MLP，不是某篇论文的专有名词，而是「只能看最近 N 步」的传统时序模型写法。
    当 gap 足够大时，首 token（A/B）落在窗口外，模型在信息论意义上不可能做对，准确率应接近 50%。
    """

    def __init__(self, vocab_size: int, history_window: int, hidden_size: int) -> None:
        super().__init__()
        self.history_window = history_window
        self.net = nn.Sequential(
            nn.Linear(vocab_size * history_window, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        visible_tokens = token_ids[:, -self.history_window :]
        features = one_hot(visible_tokens, len(VOCAB)).reshape(token_ids.size(0), -1)
        return self.net(features)


class RNNClassifier(nn.Module):
    """整条序列逐步输入 RNN；用最后时刻隐状态分类。"""

    def __init__(self, vocab_size: int, hidden_size: int, num_classes: int = 2) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence = one_hot(token_ids, len(VOCAB))
        outputs, _ = self.rnn(sequence)
        return self.head(outputs[:, -1, :])


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_classes: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence = one_hot(token_ids, len(VOCAB))
        outputs, _ = self.lstm(sequence)
        return self.head(outputs[:, -1, :])


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def lstm_hidden_for_matched_params(
    vocab_size: int,
    rnn_hidden: int,
    num_classes: int,
    hi: int = 512,
) -> int:
    """使 LSTM 总参数量尽可能接近同 vocab/num_classes 下 RNN(rnn_hidden) 的总参数量。"""
    target = count_trainable_params(RNNClassifier(vocab_size, rnn_hidden, num_classes))
    best_h, best_diff = 1, float("inf")
    for h in range(1, hi + 1):
        n = count_trainable_params(LSTMClassifier(vocab_size, h, num_classes))
        d = abs(n - target)
        if d < best_diff:
            best_diff, best_h = d, h
    return best_h


def build_model(
    model_name: str,
    config: Config,
    seq_len: int,
    num_classes: int = 2,
    lstm_hidden_size: int | None = None,
) -> nn.Module:
    if model_name == "History-MLP":
        return FixedWindowMLP(len(VOCAB), config.history_window, config.hidden_size)
    if model_name == "Flat-MLP":
        return FlatSequenceMLP(len(VOCAB), seq_len, config.hidden_size)
    if model_name == "RNN":
        return RNNClassifier(len(VOCAB), config.hidden_size, num_classes)
    if model_name == "LSTM":
        h = lstm_hidden_size if lstm_hidden_size is not None else config.hidden_size
        return LSTMClassifier(len(VOCAB), h, num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        logits = model(sequences)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item() * sequences.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader.dataset), correct / total


def run_single_experiment(
    gap_length: int,
    model_name: str,
    config: Config,
    seed: int,
    data_dir: Path | None = None,
    *,
    prefix_len: int | None = None,
    phase2_files_root: Path | None = None,
    num_classes: int = 2,
    lstm_hidden_override: int | None = None,
) -> dict[str, float | list[float] | int | None]:
    """prefix_len 为 None 时：单 token 延迟记忆（Phase 1 全模型）。非 None 时：Phase 2 多 token 任务。"""
    reseed(seed)

    if prefix_len is None:
        seq_len = gap_length + 2
        if data_dir is not None:
            gap_path = data_dir / f"gap_{gap_length:03d}"
            train_dataset: Dataset = DelayedMemoryDatasetFromFiles(gap_path / "train.jsonl")
            test_dataset = DelayedMemoryDatasetFromFiles(gap_path / "test.jsonl")
        else:
            train_dataset = DelayedMemoryDataset(gap_length, config.train_samples_per_gap, seed + 100)
            test_dataset = DelayedMemoryDataset(gap_length, config.test_samples_per_gap, seed + 200)
    else:
        seq_len = prefix_len + gap_length + 1
        if phase2_files_root is not None:
            gap_path = phase2_files_root / f"phase2_k{prefix_len}" / f"gap_{gap_length:03d}"
            train_dataset = DelayedMemoryDatasetFromFiles(
                gap_path / "train.jsonl", expected_prefix_len=prefix_len
            )
            test_dataset = DelayedMemoryDatasetFromFiles(
                gap_path / "test.jsonl", expected_prefix_len=prefix_len
            )
        else:
            train_dataset = MultiTokenDelayedMemoryDataset(
                prefix_len, gap_length, config.train_samples_per_gap, seed + 100
            )
            test_dataset = MultiTokenDelayedMemoryDataset(
                prefix_len, gap_length, config.test_samples_per_gap, seed + 200
            )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    lstm_h: int | None = None
    if model_name == "LSTM" and lstm_hidden_override is not None:
        lstm_h = lstm_hidden_override

    model = build_model(model_name, config, seq_len, num_classes=num_classes, lstm_hidden_size=lstm_h).to(
        config.device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    epochs_to_threshold: int | None = None
    seconds_to_threshold: float | None = None
    cumulative_seconds = 0.0

    phase_tag = f"p2_k{prefix_len}" if prefix_len is not None else "p1"
    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        cumulative_seconds += time.perf_counter() - t0
        train_losses.append(train_loss)

        if epochs_to_threshold is None and train_loss < config.train_loss_threshold:
            epochs_to_threshold = epoch
            seconds_to_threshold = cumulative_seconds

        if seed == config.seeds[0] and epoch in (1, config.epochs):
            _, test_accuracy = evaluate(model, test_loader, criterion, config.device)
            print(
                f"[{phase_tag}] [gap={gap_length:>3}] [{model_name}] [seed={seed}] "
                f"epoch={epoch:02d} train_loss={train_loss:.4f} test_acc={test_accuracy:.4f}"
            )

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, config.device)
    return {
        "gap_length": gap_length,
        "seed": seed,
        "train_losses": train_losses,
        "final_test_loss": test_loss,
        "final_test_accuracy": test_accuracy,
        "epochs_to_threshold": epochs_to_threshold,
        "seconds_to_threshold": seconds_to_threshold,
    }


def aggregate_seed_runs(seed_runs: list[dict[str, float | list[float]]]) -> dict[str, float | list[float]]:
    accuracies = [float(run["final_test_accuracy"]) for run in seed_runs]
    losses = [float(run["final_test_loss"]) for run in seed_runs]
    training_curves = np.array([run["train_losses"] for run in seed_runs], dtype=np.float64)
    epochs_to_threshold = [run["epochs_to_threshold"] for run in seed_runs]
    seconds_to_threshold = [run["seconds_to_threshold"] for run in seed_runs]

    valid_epochs = [int(x) for x in epochs_to_threshold if x is not None]
    valid_seconds = [float(x) for x in seconds_to_threshold if x is not None]

    return {
        "seed_accuracies": accuracies,
        "mean_test_accuracy": float(np.mean(accuracies)),
        "std_test_accuracy": float(np.std(accuracies)),
        "mean_test_loss": float(np.mean(losses)),
        "mean_train_losses": training_curves.mean(axis=0).tolist(),
        "std_train_losses": training_curves.std(axis=0).tolist(),
        "seed_epochs_to_threshold": epochs_to_threshold,
        "seed_seconds_to_threshold": seconds_to_threshold,
        "mean_epochs_to_threshold": (float(np.mean(valid_epochs)) if valid_epochs else None),
        "mean_seconds_to_threshold": (float(np.mean(valid_seconds)) if valid_seconds else None),
    }


def compute_critical_gap(
    model_results: dict[int, dict[str, float | list[float]]],
    accuracy_threshold: float,
) -> int | None:
    for gap in sorted(model_results.keys()):
        if float(model_results[gap]["mean_test_accuracy"]) < accuracy_threshold:
            return gap
    return None


def summarize_critical_gaps(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    accuracy_threshold: float,
) -> dict[str, int | None]:
    return {
        model_name: compute_critical_gap(model_results, accuracy_threshold)
        for model_name, model_results in results.items()
    }


def plot_accuracy_bands(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    output_path: Path,
    title: str = "Accuracy vs Dependency Length (Mean ± Std Across Seeds)",
) -> None:
    plt.figure(figsize=(11, 6))

    for model_name, model_results in results.items():
        gap_lengths = sorted(model_results.keys())
        means = np.array([model_results[gap]["mean_test_accuracy"] for gap in gap_lengths], dtype=float)
        stds = np.array([model_results[gap]["std_test_accuracy"] for gap in gap_lengths], dtype=float)

        plt.plot(gap_lengths, means, marker="o", linewidth=2.5, label=model_name)
        plt.fill_between(gap_lengths, means - stds, means + stds, alpha=0.18)

    plt.xlabel("Dependency Gap Length")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.ylim(0.45, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def ordered_model_names_for_heatmap(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    row_order: tuple[str, ...] = HEATMAP_ROW_ORDER,
) -> list[str]:
    ordered = [m for m in row_order if m in results]
    extra = [m for m in results if m not in row_order]
    return ordered + sorted(extra)


def plot_accuracy_heatmap(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    output_path: Path,
    plot_title: str = "Performance Heatmap",
) -> None:
    model_names = ordered_model_names_for_heatmap(results)
    gap_lengths = sorted(next(iter(results.values())).keys())
    matrix = np.array(
        [[results[model_name][gap]["mean_test_accuracy"] for gap in gap_lengths] for model_name in model_names],
        dtype=float,
    )

    plt.figure(figsize=(10, 4.8))
    im = plt.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.45, vmax=1.0)
    plt.colorbar(im, label="Mean Test Accuracy")
    plt.xticks(np.arange(len(gap_lengths)), gap_lengths)
    plt.yticks(np.arange(len(model_names)), model_names)
    plt.xlabel("Dependency Gap Length")
    plt.title(plot_title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_seed_scatter(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    output_path: Path,
    plot_title: str = "Seed-Level Accuracy Distribution",
) -> None:
    model_names = list(results.keys())
    colors = {
        "History-MLP": "#4C78A8",
        "Flat-MLP": "#B279A2",
        "RNN": "#F58518",
        "LSTM": "#54A24B",
    }

    plt.figure(figsize=(11, 6))

    for model_name in model_names:
        color = colors.get(model_name, "#333333")
        for gap, metrics in results[model_name].items():
            xs = np.full(len(metrics["seed_accuracies"]), gap, dtype=float)
            jitter = np.linspace(-0.5, 0.5, len(xs))
            plt.scatter(
                xs + jitter,
                metrics["seed_accuracies"],
                alpha=0.7,
                s=36,
                color=color,
                label=model_name if gap == min(results[model_name].keys()) else None,
            )

    plt.xlabel("Dependency Gap Length")
    plt.ylabel("Seed-Level Test Accuracy")
    plt.title(plot_title)
    plt.ylim(0.45, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_training_snapshot(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    selected_gap: int,
    output_path: Path,
    plot_title: str | None = None,
    epoch_start: int = 1,
    exclude_models: frozenset[str] | None = None,
) -> None:
    """epoch_start：从第几个 epoch 开始画（1-based），用于放大后期变化。exclude_models：不画某些模型（如 Flat-MLP 压扁纵轴）。"""

    plt.figure(figsize=(10, 6))

    epoch_end: int | None = None
    for model_name, model_results in results.items():
        if exclude_models and model_name in exclude_models:
            continue
        means = np.array(model_results[selected_gap]["mean_train_losses"], dtype=float)
        stds = np.array(model_results[selected_gap]["std_train_losses"], dtype=float)
        if epoch_start > 1:
            i0 = epoch_start - 1
            if i0 >= len(means):
                continue
            means = means[i0:]
            stds = stds[i0:]
        epochs = np.arange(epoch_start, epoch_start + len(means))
        epoch_end = epoch_start + len(means) - 1

        plt.plot(epochs, means, linewidth=2.3, label=model_name)
        plt.fill_between(epochs, means - stds, means + stds, alpha=0.18)

    if plot_title is None:
        if epoch_end is not None:
            plot_title = f"Training Loss at Gap = {selected_gap} (epochs {epoch_start}–{epoch_end}, Mean ± Std)"
        else:
            plot_title = f"Training Loss at Gap = {selected_gap} (Mean ± Std)"
    elif epoch_start > 1 and epoch_end is not None and "epochs" not in plot_title.lower():
        plot_title = f"{plot_title} (epochs {epoch_start}–{epoch_end})"

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(plot_title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_time_to_threshold(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    output_path: Path,
    plot_title: str,
) -> None:
    plt.figure(figsize=(11, 6))

    for model_name, model_results in results.items():
        gap_lengths = sorted(model_results.keys())
        values = []
        for gap in gap_lengths:
            mean_epochs = model_results[gap]["mean_epochs_to_threshold"]
            values.append(np.nan if mean_epochs is None else float(mean_epochs))

        plt.plot(gap_lengths, values, marker="o", linewidth=2.5, label=model_name)

    plt.xlabel("Dependency Gap Length")
    plt.ylabel("Mean Epochs To Loss Threshold")
    plt.title(plot_title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_metrics(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    config: Config,
    output_path: Path,
) -> None:
    serializable = {
        "config": asdict(config),
        "results": {
            model_name: {str(gap): metrics for gap, metrics in model_results.items()}
            for model_name, model_results in results.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def write_summary(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    output_path: Path,
    heading: str = "Delayed memory experiment summary",
    accuracy_threshold: float | None = None,
) -> None:
    lines = [heading, ""]

    if accuracy_threshold is not None:
        lines.append(f"Critical gap threshold: accuracy < {accuracy_threshold:.2f}")
        lines.append("")

    for model_name, model_results in results.items():
        lines.append(model_name)
        if accuracy_threshold is not None:
            critical_gap = compute_critical_gap(model_results, accuracy_threshold)
            lines.append(f"  critical_gap: {critical_gap}")
        for gap in sorted(model_results.keys()):
            mean_acc = model_results[gap]["mean_test_accuracy"]
            std_acc = model_results[gap]["std_test_accuracy"]
            mean_epochs = model_results[gap]["mean_epochs_to_threshold"]
            epoch_text = "None" if mean_epochs is None else f"{mean_epochs:.2f}"
            lines.append(
                f"  gap={gap:>3}: mean={mean_acc:.4f}, std={std_acc:.4f}, epochs_to_threshold={epoch_text}"
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def pick_models(
    results: dict[str, dict[int, dict[str, float | list[float]]]],
    model_names: list[str],
) -> dict[str, dict[int, dict[str, float | list[float]]]]:
    return {name: results[name] for name in model_names if name in results}


def warn_if_narrow_gap_before_training(gap_lengths: tuple[int, ...]) -> None:
    """在长时间训练前提示：仅小 gap 时 Phase2 曲线易重叠。"""
    if max(gap_lengths) >= RECOMMENDED_GAP_MAX_FOR_PHASE2:
        return
    print(
        f"\n[提示] 当前 max(gap)={max(gap_lengths)}，Phase2 里 RNN 与 LSTM 在短依赖上常都接近 100%，图线会贴在一起。\n"
        "若要做「LSTM 更稳」的对比，请加长大 gap，例如：python experiment.py --preset slides\n",
        flush=True,
    )


def warn_if_gap_regime_too_short_for_phase2_story(
    config: Config,
    phase2: dict[str, dict[int, dict[str, float | list[float]]]],
) -> None:
    """长 gap 上 RNN/LSTM 才容易拉开；仅小 gap 时两曲线会贴在一起，容易被误认为「没效果」。"""
    max_gap = max(config.gap_lengths)
    if max_gap >= 18:
        return

    rnn_ok = all(phase2["RNN"][g]["mean_test_accuracy"] >= 0.95 for g in config.gap_lengths)
    lstm_ok = all(phase2["LSTM"][g]["mean_test_accuracy"] >= 0.95 for g in config.gap_lengths)
    if rnn_ok and lstm_ok:
        print(
            "\n[提示] 当前配置的 max gap 较小，RNN 与 LSTM 在测试集上往往都接近 100%，"
            "phase2 图线会重叠，这是正常现象，不是数据或代码坏了。\n"
            "若要体现「LSTM 更稳 / 长依赖更难」的对比，请加大 gap，例如使用默认：\n"
            "  python experiment.py --preset slides\n"
        )


def build_matched_parameter_rows(config: Config) -> list[dict[str, int]]:
    num_classes = 2**config.phase2_prefix_len
    seq_len = config.phase2_prefix_len + max(config.gap_lengths) + 1

    history = build_model("History-MLP", config, seq_len, num_classes=2)
    flat = build_model("Flat-MLP", config, seq_len, num_classes=2)
    rnn = build_model("RNN", config, seq_len, num_classes=num_classes)

    if config.match_lstm_params:
        lstm_hidden = lstm_hidden_for_matched_params(len(VOCAB), config.hidden_size, num_classes)
    else:
        lstm_hidden = config.hidden_size

    lstm = build_model("LSTM", config, seq_len, num_classes=num_classes, lstm_hidden_size=lstm_hidden)

    return [
        {"model": "History-MLP", "hidden_size": config.hidden_size, "params": count_trainable_params(history)},
        {"model": "Flat-MLP", "hidden_size": config.hidden_size, "params": count_trainable_params(flat)},
        {"model": "RNN", "hidden_size": config.hidden_size, "params": count_trainable_params(rnn)},
        {"model": "LSTM", "hidden_size": lstm_hidden, "params": count_trainable_params(lstm)},
    ]


def write_parameter_table(rows: list[dict[str, int]], output_path: Path) -> None:
    lines = [
        "| Model | Hidden Size | Trainable Params |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['hidden_size']} | {row['params']} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_cli_preset(args)
    config = Config(
        gap_lengths=tuple(args.gaps),
        seeds=tuple(args.seeds),
        train_samples_per_gap=args.train_samples,
        test_samples_per_gap=args.test_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        training_loss_epoch_start=args.training_loss_epoch_start,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        history_window=args.history_window,
        phase2_prefix_len=args.phase2_prefix_len,
        acc_threshold_critical_gap=args.acc_threshold_critical_gap,
        train_loss_threshold=args.train_loss_threshold,
        match_lstm_params=args.match_lstm_params,
        phase2_data_dir=args.phase2_data_dir if args.phase2_data_dir is not None else args.data_dir,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.export_data:
        export_delayed_memory_jsonl(config, args.data_export_root)
        export_phase2_multi_token_jsonl(config, args.data_export_root)
        print(f"Exported JSONL datasets to: {args.data_export_root.resolve()}")

    data_dir: Path | None = args.data_dir
    if data_dir is not None:
        print(
            "Using fixed data from disk (--data-dir): train/test are identical across seeds; "
            "variance reflects initialization and batch order only."
        )

    print(
        f"Config: gap_lengths={list(config.gap_lengths)}, seeds={list(config.seeds)}, "
        f"epochs={config.epochs}, training_loss_epoch_start={config.training_loss_epoch_start}, "
        f"phase2_prefix_len={config.phase2_prefix_len}, device={config.device}",
        flush=True,
    )
    warn_if_narrow_gap_before_training(config.gap_lengths)

    phase1_models = ["History-MLP", "Flat-MLP", "RNN"]
    phase2_models = ["RNN", "LSTM"]
    phase1: dict[str, dict[int, dict[str, float | list[float]]]] = {name: {} for name in phase1_models}
    phase2: dict[str, dict[int, dict[str, float | list[float]]]] = {name: {} for name in phase2_models}

    phase2_num_classes = 2**config.phase2_prefix_len
    phase2_lstm_hidden = (
        lstm_hidden_for_matched_params(len(VOCAB), config.hidden_size, phase2_num_classes)
        if config.match_lstm_params
        else config.hidden_size
    )

    for gap_length in config.gap_lengths:
        print(f"\n=== Phase 1 gap length: {gap_length} ===")
        for model_name in phase1_models:
            seed_runs = []
            for seed in config.seeds:
                seed_runs.append(run_single_experiment(gap_length, model_name, config, seed, data_dir=data_dir))
            phase1[model_name][gap_length] = aggregate_seed_runs(seed_runs)

    for gap_length in config.gap_lengths:
        print(f"\n=== Phase 2 gap length: {gap_length} ===")
        for model_name in phase2_models:
            seed_runs = []
            for seed in config.seeds:
                seed_runs.append(
                    run_single_experiment(
                        gap_length,
                        model_name,
                        config,
                        seed,
                        prefix_len=config.phase2_prefix_len,
                        phase2_files_root=config.phase2_data_dir,
                        num_classes=phase2_num_classes,
                        lstm_hidden_override=(phase2_lstm_hidden if model_name == "LSTM" else None),
                    )
                )
            phase2[model_name][gap_length] = aggregate_seed_runs(seed_runs)

    results = {
        "History-MLP": phase1["History-MLP"],
        "Flat-MLP": phase1["Flat-MLP"],
        "RNN": phase2["RNN"],
        "LSTM": phase2["LSTM"],
    }

    plot_accuracy_bands(
        phase1,
        OUTPUT_DIR / "phase1_accuracy_vs_gap_bands.png",
        title="Phase 1: Fixed window & flat MLP vs RNN (Mean ± Std)",
    )
    plot_accuracy_heatmap(phase1, OUTPUT_DIR / "phase1_accuracy_heatmap.png", plot_title="Phase 1: Heatmap")
    plot_seed_scatter(phase1, OUTPUT_DIR / "phase1_seed_accuracy_scatter.png", plot_title="Phase 1: Seed-Level Accuracy")
    plot_training_snapshot(
        phase1,
        selected_gap=max(config.gap_lengths),
        output_path=OUTPUT_DIR / "phase1_training_loss_gap_max.png",
        plot_title=f"Phase 1: Training Loss at Gap = {max(config.gap_lengths)} (Mean ± Std)",
        epoch_start=config.training_loss_epoch_start,
    )
    plot_time_to_threshold(
        phase1,
        OUTPUT_DIR / "phase1_time_to_threshold.png",
        plot_title="Phase 1: Mean Epochs To Reach Loss Threshold",
    )

    plot_accuracy_bands(
        phase2,
        OUTPUT_DIR / "phase2_accuracy_vs_gap_bands.png",
        title=f"Phase 2: Multi-token memory (k={config.phase2_prefix_len}) — RNN vs LSTM (Mean ± Std)",
    )
    plot_accuracy_heatmap(phase2, OUTPUT_DIR / "phase2_accuracy_heatmap.png", plot_title="Phase 2: Heatmap")
    plot_seed_scatter(phase2, OUTPUT_DIR / "phase2_seed_accuracy_scatter.png", plot_title="Phase 2: Seed-Level Accuracy")
    plot_training_snapshot(
        phase2,
        selected_gap=max(config.gap_lengths),
        output_path=OUTPUT_DIR / "phase2_training_loss_gap_max.png",
        plot_title=f"Phase 2: Training Loss at Gap = {max(config.gap_lengths)} (Mean ± Std)",
        epoch_start=config.training_loss_epoch_start,
    )
    plot_time_to_threshold(
        phase2,
        OUTPUT_DIR / "phase2_time_to_threshold.png",
        plot_title="Phase 2: Mean Epochs To Reach Loss Threshold",
    )

    plot_accuracy_bands(
        results,
        OUTPUT_DIR / "accuracy_vs_gap_bands.png",
        title="Two-phase overview: baseline history models, RNN, and LSTM",
    )
    plot_accuracy_heatmap(results, OUTPUT_DIR / "accuracy_heatmap.png", plot_title="All Models: Heatmap")
    plot_seed_scatter(results, OUTPUT_DIR / "seed_accuracy_scatter.png", plot_title="All Models: Seed-Level Accuracy")
    plot_training_snapshot(
        results,
        selected_gap=max(config.gap_lengths),
        output_path=OUTPUT_DIR / "training_loss_gap_max.png",
        plot_title=f"History-MLP, RNN, LSTM (excl. Flat-MLP): Training Loss at Gap = {max(config.gap_lengths)} (Mean ± Std)",
        epoch_start=config.training_loss_epoch_start,
        exclude_models=frozenset({"Flat-MLP"}),
    )

    save_metrics(phase1, config, OUTPUT_DIR / "phase1_metrics.json")
    save_metrics(phase2, config, OUTPUT_DIR / "phase2_metrics.json")
    save_metrics(results, config, OUTPUT_DIR / "metrics.json")

    phase2_critical_gaps = summarize_critical_gaps(phase2, config.acc_threshold_critical_gap)
    parameter_rows = build_matched_parameter_rows(config)

    combined = {
        "config": asdict(config),
        "phase1_models": phase1_models,
        "phase2_models": phase2_models,
        "phase2_num_classes": phase2_num_classes,
        "phase2_critical_gaps": phase2_critical_gaps,
        "matched_parameter_rows": parameter_rows,
        "phase1": {
            model_name: {str(gap): metrics for gap, metrics in phase1[model_name].items()} for model_name in phase1
        },
        "phase2": {
            model_name: {str(gap): metrics for gap, metrics in phase2[model_name].items()} for model_name in phase2
        },
        "all_models": {
            model_name: {str(gap): metrics for gap, metrics in results[model_name].items()} for model_name in results
        },
    }
    with open(OUTPUT_DIR / "metrics_combined.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    write_summary(
        phase1,
        OUTPUT_DIR / "phase1_summary.txt",
        heading="Phase 1: History-MLP, Flat-MLP, RNN",
        accuracy_threshold=config.acc_threshold_critical_gap,
    )
    write_summary(
        phase2,
        OUTPUT_DIR / "phase2_summary.txt",
        heading=f"Phase 2: multi-token delayed memory (k={config.phase2_prefix_len}) — RNN vs LSTM",
        accuracy_threshold=config.acc_threshold_critical_gap,
    )
    write_summary(
        results,
        OUTPUT_DIR / "summary.txt",
        heading="All models summary",
        accuracy_threshold=config.acc_threshold_critical_gap,
    )
    write_parameter_table(parameter_rows, OUTPUT_DIR / "matched_parameter_table.md")

    warn_if_gap_regime_too_short_for_phase2_story(config, phase2)

    print(f"\nArtifacts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

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


DEFAULT_OUTPUT_DIR = Path("outputs_exp2_copy_task")


def reseed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Config:
    gap_lengths: tuple[int, ...] = (10, 20, 30, 40)
    seeds: tuple[int, ...] = (1, 2, 3)
    memory_length: int = 8
    train_samples_per_gap: int = 2500
    test_samples_per_gap: int = 500
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    embedding_dim: int = 32
    hidden_size: int = 64
    num_symbols: int = 8
    acc_threshold_critical_gap: float = 0.80
    tail_loss_only: bool = True
    token_threshold: float = 0.60
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def blank_token(self) -> int:
        return 0

    @property
    def delimiter_token(self) -> int:
        return self.num_symbols + 1

    @property
    def vocab_size(self) -> int:
        return self.num_symbols + 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2: copy-memory task for RNN vs LSTM")
    parser.add_argument("--gaps", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--memory-length", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=2500)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-symbols", type=int, default=8)
    parser.add_argument("--acc-threshold", type=float, default=0.80)
    parser.add_argument("--token-threshold", type=float, default=0.60)
    parser.add_argument("--full-sequence-loss", action="store_true")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


class CopyMemoryDataset(Dataset):
    def __init__(self, gap_length: int, sample_count: int, config: Config, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.memory_length = config.memory_length
        self.sequence_length = config.memory_length + gap_length + 1 + config.memory_length

        inputs = []
        targets = []
        for _ in range(sample_count):
            memory = rng.integers(1, config.num_symbols + 1, size=config.memory_length, dtype=np.int64)
            prefix_blanks = np.full(gap_length, config.blank_token, dtype=np.int64)
            suffix_blanks = np.full(config.memory_length, config.blank_token, dtype=np.int64)

            sequence = np.concatenate(
                [
                    memory,
                    prefix_blanks,
                    np.array([config.delimiter_token], dtype=np.int64),
                    suffix_blanks,
                ]
            )
            target = np.concatenate(
                [
                    np.full(config.memory_length + gap_length + 1, config.blank_token, dtype=np.int64),
                    memory,
                ]
            )

            inputs.append(sequence)
            targets.append(target)

        self.inputs = torch.tensor(np.stack(inputs), dtype=torch.long)
        self.targets = torch.tensor(np.stack(targets), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class CopyRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        del vocab_size
        sequence = self.embedding(token_ids)
        outputs, _ = self.rnn(sequence)
        return self.head(outputs)


class CopyLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        del vocab_size
        sequence = self.embedding(token_ids)
        outputs, _ = self.lstm(sequence)
        return self.head(outputs)


def build_model(model_name: str, config: Config) -> nn.Module:
    if model_name == "RNN":
        return CopyRNN(config.vocab_size, config.embedding_dim, config.hidden_size)
    if model_name == "LSTM":
        return CopyLSTM(config.vocab_size, config.embedding_dim, config.hidden_size)
    raise ValueError(f"Unknown model: {model_name}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: Config,
) -> float:
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)

        optimizer.zero_grad()
        logits = model(inputs, config.vocab_size)
        if config.tail_loss_only:
            logits_for_loss = logits[:, -config.memory_length :, :]
            targets_for_loss = targets[:, -config.memory_length :]
        else:
            logits_for_loss = logits
            targets_for_loss = targets
        loss = criterion(logits_for_loss.reshape(-1, config.vocab_size), targets_for_loss.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, config: Config) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    token_correct = 0
    token_total = 0
    sequence_correct = 0
    sequence_total = 0

    for inputs, targets in loader:
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        logits = model(inputs, config.vocab_size)

        if config.tail_loss_only:
            logits_for_loss = logits[:, -config.memory_length :, :]
            targets_for_loss = targets[:, -config.memory_length :]
        else:
            logits_for_loss = logits
            targets_for_loss = targets

        loss = nn.functional.cross_entropy(
            logits_for_loss.reshape(-1, config.vocab_size),
            targets_for_loss.reshape(-1),
        )
        total_loss += loss.item() * inputs.size(0)

        predictions = torch.argmax(logits, dim=-1)
        target_tail = targets[:, -config.memory_length :]
        prediction_tail = predictions[:, -config.memory_length :]

        token_correct += (prediction_tail == target_tail).sum().item()
        token_total += target_tail.numel()
        sequence_correct += (prediction_tail == target_tail).all(dim=1).sum().item()
        sequence_total += target_tail.size(0)

    return (
        total_loss / len(loader.dataset),
        token_correct / token_total,
        sequence_correct / sequence_total,
    )


def run_single_experiment(gap_length: int, model_name: str, config: Config, seed: int) -> dict[str, float | list[float] | None]:
    reseed(seed)

    train_dataset = CopyMemoryDataset(gap_length, config.train_samples_per_gap, config, seed + 100)
    test_dataset = CopyMemoryDataset(gap_length, config.test_samples_per_gap, config, seed + 200)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = build_model(model_name, config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    token_accuracies = []
    sequence_accuracies = []
    epochs_to_loss_threshold = None
    cumulative_seconds = 0.0
    seconds_to_loss_threshold = None
    epochs_to_token_threshold = None
    seconds_to_token_threshold = None

    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config)
        cumulative_seconds += time.perf_counter() - start
        train_losses.append(train_loss)

        test_loss, token_acc, seq_acc = evaluate(model, test_loader, config)
        token_accuracies.append(token_acc)
        sequence_accuracies.append(seq_acc)

        if epochs_to_loss_threshold is None and train_loss < 0.2:
            epochs_to_loss_threshold = epoch
            seconds_to_loss_threshold = cumulative_seconds

        if epochs_to_token_threshold is None and token_acc >= config.token_threshold:
            epochs_to_token_threshold = epoch
            seconds_to_token_threshold = cumulative_seconds

        if seed == config.seeds[0] and epoch in (1, config.epochs):
            print(
                f"[copy] [gap={gap_length:>3}] [{model_name}] [seed={seed}] "
                f"epoch={epoch:02d} train_loss={train_loss:.4f} token_acc={token_acc:.4f} seq_acc={seq_acc:.4f}"
            )

    return {
        "gap_length": gap_length,
        "seed": seed,
        "train_losses": train_losses,
        "token_accuracies": token_accuracies,
        "sequence_accuracies": sequence_accuracies,
        "final_test_loss": test_loss,
        "final_token_accuracy": token_acc,
        "final_sequence_accuracy": seq_acc,
        "epochs_to_loss_threshold": epochs_to_loss_threshold,
        "seconds_to_loss_threshold": seconds_to_loss_threshold,
        "epochs_to_token_threshold": epochs_to_token_threshold,
        "seconds_to_token_threshold": seconds_to_token_threshold,
    }


def aggregate_seed_runs(seed_runs: list[dict[str, float | list[float] | None]]) -> dict[str, float | list[float] | None]:
    token_accs = [float(run["final_token_accuracy"]) for run in seed_runs]
    seq_accs = [float(run["final_sequence_accuracy"]) for run in seed_runs]
    losses = [float(run["final_test_loss"]) for run in seed_runs]
    train_curves = np.array([run["train_losses"] for run in seed_runs], dtype=np.float64)
    token_curves = np.array([run["token_accuracies"] for run in seed_runs], dtype=np.float64)
    seq_curves = np.array([run["sequence_accuracies"] for run in seed_runs], dtype=np.float64)
    loss_threshold_epochs = [run["epochs_to_loss_threshold"] for run in seed_runs]
    valid_loss_epochs = [int(x) for x in loss_threshold_epochs if x is not None]
    token_threshold_epochs = [run["epochs_to_token_threshold"] for run in seed_runs]
    valid_token_epochs = [int(x) for x in token_threshold_epochs if x is not None]

    return {
        "seed_token_accuracies": token_accs,
        "seed_sequence_accuracies": seq_accs,
        "mean_token_accuracy": float(np.mean(token_accs)),
        "std_token_accuracy": float(np.std(token_accs)),
        "mean_sequence_accuracy": float(np.mean(seq_accs)),
        "std_sequence_accuracy": float(np.std(seq_accs)),
        "mean_test_loss": float(np.mean(losses)),
        "mean_train_losses": train_curves.mean(axis=0).tolist(),
        "std_train_losses": train_curves.std(axis=0).tolist(),
        "mean_token_accuracies": token_curves.mean(axis=0).tolist(),
        "std_token_accuracies": token_curves.std(axis=0).tolist(),
        "mean_sequence_accuracies": seq_curves.mean(axis=0).tolist(),
        "std_sequence_accuracies": seq_curves.std(axis=0).tolist(),
        "mean_epochs_to_loss_threshold": float(np.mean(valid_loss_epochs)) if valid_loss_epochs else None,
        "mean_epochs_to_token_threshold": float(np.mean(valid_token_epochs)) if valid_token_epochs else None,
    }


def compute_critical_gap(results: dict[int, dict[str, float | list[float] | None]], threshold: float) -> int | None:
    for gap in sorted(results.keys()):
        if float(results[gap]["mean_sequence_accuracy"]) < threshold:
            return gap
    return None


def plot_sequence_accuracy(results: dict[str, dict[int, dict[str, float | list[float] | None]]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        gaps = sorted(model_results.keys())
        means = np.array([model_results[g]["mean_sequence_accuracy"] for g in gaps], dtype=float)
        stds = np.array([model_results[g]["std_sequence_accuracy"] for g in gaps], dtype=float)
        plt.plot(gaps, means, marker="o", linewidth=2.5, label=model_name)
        plt.fill_between(gaps, means - stds, means + stds, alpha=0.18)

    plt.xlabel("Gap Length")
    plt.ylabel("Exact Sequence Accuracy")
    plt.title("Experiment 2: Copy Memory Task")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_token_accuracy(results: dict[str, dict[int, dict[str, float | list[float] | None]]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        gaps = sorted(model_results.keys())
        means = np.array([model_results[g]["mean_token_accuracy"] for g in gaps], dtype=float)
        stds = np.array([model_results[g]["std_token_accuracy"] for g in gaps], dtype=float)
        plt.plot(gaps, means, marker="o", linewidth=2.5, label=model_name)
        plt.fill_between(gaps, means - stds, means + stds, alpha=0.18)

    plt.xlabel("Gap Length")
    plt.ylabel("Token Accuracy")
    plt.title("Experiment 2: Token-Level Copy Accuracy")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_learning_curve_for_gap(
    results: dict[str, dict[int, dict[str, float | list[float] | None]]],
    gap: int,
    metric_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        if gap not in model_results:
            continue
        means = np.array(model_results[gap][metric_key], dtype=float)
        stds = np.array(model_results[gap][std_key], dtype=float)
        epochs = np.arange(1, len(means) + 1)
        plt.plot(epochs, means, linewidth=2.5, label=model_name)
        plt.fill_between(epochs, means - stds, means + stds, alpha=0.18)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_summary(results: dict[str, dict[int, dict[str, float | list[float] | None]]], config: Config, output_path: Path) -> None:
    lines = [
        "Experiment 2: copy-memory task",
        f"Critical gap threshold: exact sequence accuracy < {config.acc_threshold_critical_gap:.2f}",
        "",
    ]
    for model_name, model_results in results.items():
        lines.append(model_name)
        lines.append(f"  critical_gap: {compute_critical_gap(model_results, config.acc_threshold_critical_gap)}")
        for gap in sorted(model_results.keys()):
            lines.append(
                f"  gap={gap:>3}: seq_mean={model_results[gap]['mean_sequence_accuracy']:.4f}, "
                f"seq_std={model_results[gap]['std_sequence_accuracy']:.4f}, "
                f"token_mean={model_results[gap]['mean_token_accuracy']:.4f}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = Config(
        gap_lengths=tuple(args.gaps),
        seeds=tuple(args.seeds),
        memory_length=args.memory_length,
        train_samples_per_gap=args.train_samples,
        test_samples_per_gap=args.test_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_symbols=args.num_symbols,
        acc_threshold_critical_gap=args.acc_threshold,
        tail_loss_only=not args.full_sequence_loss,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Config: gaps={list(config.gap_lengths)}, seeds={list(config.seeds)}, memory_length={config.memory_length}, "
        f"epochs={config.epochs}, hidden={config.hidden_size}, device={config.device}"
    )

    results = {"RNN": {}, "LSTM": {}}
    for gap in config.gap_lengths:
        print(f"\n=== Gap {gap} ===")
        for model_name in results:
            runs = [run_single_experiment(gap, model_name, config, seed) for seed in config.seeds]
            results[model_name][gap] = aggregate_seed_runs(runs)

    plot_sequence_accuracy(results, output_dir / "exp2_copy_sequence_accuracy.png")
    plot_token_accuracy(results, output_dir / "exp2_copy_token_accuracy.png")
    focus_gap = config.gap_lengths[len(config.gap_lengths) // 2]
    plot_learning_curve_for_gap(
        results,
        focus_gap,
        "mean_sequence_accuracies",
        "std_sequence_accuracies",
        "Exact Sequence Accuracy",
        f"Experiment 2: Sequence Accuracy vs Epoch (gap={focus_gap})",
        output_dir / "exp2_copy_sequence_learning_curve.png",
    )
    plot_learning_curve_for_gap(
        results,
        focus_gap,
        "mean_token_accuracies",
        "std_token_accuracies",
        "Token Accuracy",
        f"Experiment 2: Token Accuracy vs Epoch (gap={focus_gap})",
        output_dir / "exp2_copy_token_learning_curve.png",
    )
    write_summary(results, config, output_dir / "exp2_copy_summary.txt")

    with open(output_dir / "exp2_copy_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(config),
                "results": {
                    model_name: {str(gap): metrics for gap, metrics in model_results.items()}
                    for model_name, model_results in results.items()
                },
            },
            f,
            indent=2,
        )

    print(f"\nArtifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

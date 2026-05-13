import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from E_pos_eval import build_pos_samples
from readers import ReadUPOSInputFile


class SequencePOS_ANN(nn.Module):
    """Feedforward MLP mirroring the SNN layer sizes (ReLU activations)."""

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_loss_and_optimizer(model: nn.Module, learning_rate: float = 5e-4, class_weights: torch.Tensor = None):
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer


def save_training_metadata(metadata_path: Path, metadata: dict):
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_pos"


def main():
    parser = argparse.ArgumentParser(description="Train an MLP for token-level POS tagging")
    # input is always spatial for the MLP (flattened context window)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input_file_prefix", type=str, default="pos_d100")
    parser.add_argument("--output_file_prefix", type=str, default="upos-win-ann-mlp")
    parser.add_argument("--num_hidden_1", type=int, default=256)
    parser.add_argument("--num_hidden_2", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "output_results" / "E_pos" / "win-mlp"))
    args = parser.parse_args()

    # MLP uses flattened spatial windows only

    sent_train_data, embedding_dim = ReadUPOSInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_train.pkl", limit=args.limit)
    sent_test_data, _ = ReadUPOSInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_test.pkl", limit=args.limit)

    def collect_pos_tags(sentences):
        tags = set()
        for sentence in sentences:
            for token in sentence:
                if len(token) > 1:
                    tags.add(token[1])
        return sorted(tags)

    pos_tags = collect_pos_tags(sent_train_data + sent_test_data)
    label_to_idx = {tag: i for i, tag in enumerate(pos_tags)}
    idx_to_label = {i: tag for tag, i in label_to_idx.items()}

    X_train, y_train = build_pos_samples(sent_train_data, embedding_dim, label_to_idx, window_size=5)
    X_test, y_test = build_pos_samples(sent_test_data, embedding_dim, label_to_idx, window_size=5)

    # If build_pos_samples returned numpy/tensors without labels mapping, assume labels are integer tensors already.
    # X_train shape: [samples, window_len, embed_dim]
    window_len = X_train.shape[1]

    input_size = window_len * embedding_dim

    if args.limit is not None:
        train_limit = min(args.limit, X_train.shape[0])
        test_limit = min(args.limit, X_test.shape[0])
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]
        X_test = X_test[:test_limit]
        y_test = y_test[:test_limit]

    num_labels = len(label_to_idx)

    # Flatten spatial windows for MLP input
    X_train_flat = X_train.view(X_train.size(0), -1).float()
    X_test_flat = X_test.view(X_test.size(0), -1).float()

    train_ds = TensorDataset(X_train_flat, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    class_counts = torch.bincount(y_train, minlength=num_labels)
    class_weights = torch.zeros(num_labels)
    for i in range(num_labels):
        class_weights[i] = X_train.shape[0] / (num_labels * class_counts[i])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequencePOS_ANN(input_size, args.num_hidden_1, args.num_hidden_2, num_labels).to(device)

    loss_fn, optimizer = get_loss_and_optimizer(model, learning_rate=args.learning_rate, class_weights=class_weights.to(device))

    epoch_losses = []
    epoch_accuracies = []

    pprint(vars(args) | {
        "input_size": input_size,
        "num_labels": num_labels,
        "device": str(device),
        "loss": loss_fn._get_name(),
        "optimizer": optimizer.__class__.__name__,
        "class_balance": {idx_to_label[i]: int(class_counts[i]) for i in range(num_labels)},
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "task": "window-ctx-upos-ann-mlp",
    })

    training_start_time = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_start = time.perf_counter()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

        epoch_loss = running_loss / max(1, running_total)
        epoch_acc = running_correct / max(1, running_total)
        epoch_losses.append(float(epoch_loss))
        epoch_accuracies.append(float(epoch_acc))

        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch {epoch+1}/{args.epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | time_s: {epoch_time:.2f}")

    print("Training finished.")

    metadata = {
        "training_config": {
            "training_start_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": "token_level_pos_ann",
            "embedding_dim": int(embedding_dim),
            "window_len": int(window_len),
            "input_size": int(input_size),
            "num_labels": int(num_labels),
            "device": str(device),
            "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "results": {
            "epoch_train_loss": epoch_losses,
            "epoch_train_accuracy": epoch_accuracies,
            "test_loss": None,
            "test_accuracy": None,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_base = "_".join([args.output_file_prefix or "pos_ann", now, f"e-{args.epochs}"])
    metadata_path = output_dir / f"{run_base}.json"
    save_training_metadata(metadata_path, metadata)
    print(f"Metadata saved to {metadata_path}")

    if args.save:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_class": "SequencePOS_ANN",
            "model_config": {
                "input_size": input_size,
                "hidden_size_1": args.num_hidden_1,
                "hidden_size_2": args.num_hidden_2,
                "output_size": num_labels,
            },
        }
        model_path = output_dir / f"{run_base}.pt"
        torch.save(ckpt, model_path)
        print(f"Model saved to {model_path}")

    if args.eval:
        model.eval()
        with torch.no_grad():
            X_test_flat = X_test.view(X_test.size(0), -1).float().to(device)
            y_test = y_test.to(device)
            logits = model(X_test_flat)
            test_loss = loss_fn(logits, y_test).item()
            test_acc = (logits.argmax(dim=1) == y_test).float().mean().item()
            metadata["results"]["test_loss"] = test_loss
            metadata["results"]["test_accuracy"] = test_acc
            save_training_metadata(metadata_path, metadata)
            print(f"Eval -> loss: {test_loss:.4f} | acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()

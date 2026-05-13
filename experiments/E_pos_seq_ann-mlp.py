import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from readers import ReadUPOSInputFile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_pos"


class SequencePOS_SEQ_ANN(nn.Module):
    """Token-level MLP matching the SNN per-token feedforward layers."""

    def __init__(self, emb_dim, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_seq_samples(sentences, embedding_dim, label_to_idx, max_len=10):
    samples = []
    labels = []
    masks = []

    for sentence in sentences:
        if len(sentence) > max_len:
            continue

        seq = []
        lab = []
        mask = []
        for token in sentence:
            seq.append(token[3:])
            lab.append(label_to_idx.get(token[1], 0))
            mask.append(True)

        while len(seq) < max_len:
            seq.append([0.0] * embedding_dim)
            lab.append(0)
            mask.append(False)

        samples.append(seq)
        labels.append(lab)
        masks.append(mask)

    X = torch.tensor(samples, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    mask = torch.tensor(masks, dtype=torch.bool)
    return X, y, mask


def collect_pos_tags(sentences):
    tags = set()
    for sentence in sentences:
        for token in sentence:
            if len(token) > 1:
                tags.add(token[1])
    return sorted(tags)


def save_training_metadata(metadata_path, metadata):
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train a class-balanced MLP seq2seq POS tagger")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input_file_prefix", type=str, default="pos_d100")
    parser.add_argument("--output_file_prefix", type=str, default="upos-seq-ann-mlp")
    parser.add_argument("--num_hidden_1", type=int, default=256)
    parser.add_argument("--num_hidden_2", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "output_results" / "E_pos" / "seq-mlp"))
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided")

    sent_train_data, embedding_dim = ReadUPOSInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_train.pkl", limit=None)
    sent_test_data, _ = ReadUPOSInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_test.pkl", limit=None)

    max_seq_len = max((len(sentence) for sentence in sent_test_data), default=0)
    if max_seq_len <= 0:
        raise ValueError("Unable to derive max sequence length from the test set")

    filtered_train = [sentence for sentence in sent_train_data if len(sentence) <= max_seq_len]
    filtered_test = [sentence for sentence in sent_test_data if len(sentence) <= max_seq_len]

    pos_tags = collect_pos_tags(filtered_train + filtered_test)
    label_to_idx = {tag: i for i, tag in enumerate(pos_tags)}
    idx_to_label = {i: tag for tag, i in label_to_idx.items()}
    num_labels = len(label_to_idx)

    X_train, y_train, train_mask = build_seq_samples(filtered_train, embedding_dim, label_to_idx, max_len=max_seq_len)
    X_test, y_test, test_mask = build_seq_samples(filtered_test, embedding_dim, label_to_idx, max_len=max_seq_len)

    if args.limit is not None:
        train_limit = min(args.limit, X_train.shape[0])
        test_limit = min(args.limit, X_test.shape[0])
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]
        train_mask = train_mask[:train_limit]
        X_test = X_test[:test_limit]
        y_test = y_test[:test_limit]
        test_mask = test_mask[:test_limit]

    sequence_length = X_train.shape[1]

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"X_train shape: {X_train.shape}  # [samples, seq_len, emb_dim]")
    print(f"y_train shape: {y_train.shape}  # [samples, seq_len]")
    print(f"POS tag count: {num_labels}")
    print(f"POS tags: {pos_tags}")

    train_ds = TensorDataset(X_train, y_train, train_mask)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequencePOS_SEQ_ANN(embedding_dim, args.num_hidden_1, args.num_hidden_2, num_labels).to(device)

    valid_train_labels = y_train[train_mask]
    class_counts = torch.bincount(valid_train_labels.flatten(), minlength=num_labels)
    class_weights = torch.zeros(num_labels, dtype=torch.float32)
    for i in range(num_labels):
        class_weights[i] = valid_train_labels.numel() / (num_labels * class_counts[i])

    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_filename_base = "_".join([args.output_file_prefix or "pos_seq_ann", now, f"e-{args.epochs}"])

    training_start_date = datetime.now()
    metadata_file = output_dir / f"{run_filename_base}.json"
    training_metadata = {
        "training_config": {
            "training_start_date": training_start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "training_end_date": None,
            "training_duration_s": None,
            "task": "seq2seq_pos_ann",
            "embedding_dim": int(embedding_dim),
            "sequence_length": int(sequence_length),
            "input_size": int(embedding_dim),
            "num_labels": num_labels,
            "num_training_samples": int(X_train.shape[0]),
            "num_test_samples": int(X_test.shape[0]),
            "device": str(device),
            "total_params": total_params,
        } | {k: str(v) for k, v in vars(args).items()},
        "results": {
            "epoch_train_loss": [],
            "epoch_train_accuracy": [],
            "test_loss": None,
            "test_accuracy": None,
            "class_weights": class_weights.tolist(),
        },
    }
    save_training_metadata(metadata_file, training_metadata)

    pprint({
        **vars(args),
        **training_metadata["training_config"],
        "sequence_length": sequence_length,
        "num_labels": num_labels,
        "device": str(device),
        "loss": loss_fn._get_name(),
        "optimizer": optimizer.__class__.__name__,
        # "class_weights": class_weights.tolist(),
    })

    epoch_losses = []
    epoch_accuracies = []
    model.train()
    training_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            batch_size = xb.shape[0]
            xb_flat = xb.view(batch_size * xb.shape[1], xb.shape[2])
            logits_flat = model(xb_flat)
            emissions = logits_flat.view(batch_size, xb.shape[1], -1)

            flat_logits = emissions.reshape(-1, num_labels)
            flat_labels = yb.reshape(-1)
            flat_mask = mb.reshape(-1)
            loss = loss_fn(flat_logits[flat_mask], flat_labels[flat_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            valid_tokens = int(mb.sum().item())
            running_loss += loss.item() * valid_tokens
            preds = emissions.argmax(dim=-1)
            running_correct += int(((preds == yb) & mb).sum().item())
            running_total += valid_tokens

        epoch_loss = running_loss / max(1, running_total)
        epoch_acc = running_correct / max(1, running_total)
        epoch_losses.append(float(epoch_loss))
        epoch_accuracies.append(float(epoch_acc))

        training_metadata["results"]["epoch_train_loss"] = epoch_losses
        training_metadata["results"]["epoch_train_accuracy"] = epoch_accuracies
        save_training_metadata(metadata_file, training_metadata)

        epoch_time = time.perf_counter() - epoch_start_time
        print(f"Epoch {epoch + 1}/{args.epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | time_s: {epoch_time:.2f}")

    print("Training finished.")
    training_end_date = datetime.now()
    training_metadata["training_config"]["training_end_date"] = training_end_date.strftime("%Y-%m-%d %H:%M:%S")
    training_metadata["training_config"]["training_duration_s"] = (training_end_date - training_start_date).total_seconds()
    save_training_metadata(metadata_file, training_metadata)

    if args.save:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_class": "SequencePOS_SEQ_ANN",
            "model_config": {
                "emb_dim": embedding_dim,
                "hidden_size_1": args.num_hidden_1,
                "hidden_size_2": args.num_hidden_2,
                "output_size": num_labels,
            },
            "label_maps": {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label},
            "metrics": {
                "epoch_train_loss": epoch_losses,
                "epoch_train_accuracy": epoch_accuracies,
                "class_weights": class_weights.tolist(),
            },
            "cli_args": vars(args),
        }
        model_path = output_dir / f"{run_filename_base}.pt"
        torch.save(ckpt, model_path)
        print(f"Model saved to {model_path}")

    if args.eval:
        model.eval()
        with torch.no_grad():
            batch_size = X_test.shape[0]
            xb = X_test.to(device)
            yb = y_test.to(device)
            mb = test_mask.to(device)
            xb_flat = xb.view(batch_size * xb.shape[1], xb.shape[2])
            logits_flat = model(xb_flat)
            emissions = logits_flat.view(batch_size, xb.shape[1], -1)

            flat_logits = emissions.reshape(-1, num_labels)
            flat_labels = yb.reshape(-1)
            flat_mask = mb.reshape(-1)
            test_loss = loss_fn(flat_logits[flat_mask], flat_labels[flat_mask]).item()

            test_acc = ((emissions.argmax(dim=-1) == yb) & mb).float().sum().item() / max(1, int(mb.sum().item()))
            training_metadata["results"]["test_loss"] = test_loss
            training_metadata["results"]["test_accuracy"] = test_acc
            save_training_metadata(metadata_file, training_metadata)
            print(f"Eval -> loss: {test_loss:.4f} | acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from readers import ReadSENTInputFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAST_INPUT_DIR = PROJECT_ROOT / "input_data" / "cast_sent"


def build_sentiment_samples(samples, embedding_dim):
    x_list = []
    y_list = []

    for sample_idx, sample in enumerate(samples):
        if not isinstance(sample, (list, tuple)) or len(sample) < 2:
            raise ValueError(f"Invalid sample format at index {sample_idx}: expected [sequence_embeddings, binary_label]")

        token_embeddings = sample[0]
        label_value = sample[1]

        if not token_embeddings:
            raise ValueError(f"Empty token embedding sequence at sample index {sample_idx}")

        seq_embeddings = []
        for token_idx, token_embedding in enumerate(token_embeddings):
            embedding = torch.as_tensor(token_embedding, dtype=torch.float32)
            if embedding.ndim != 1 or embedding.numel() != embedding_dim:
                raise ValueError(
                    "Embedding dimension mismatch "
                    f"at sample {sample_idx}, token {token_idx}: expected {embedding_dim}, got {embedding.numel()}"
                )
            seq_embeddings.append(embedding)

        if label_value not in (0, 1):
            raise ValueError(f"Label must be exactly 0 or 1. Got: {label_value} at sample index {sample_idx}")

        x_list.append(torch.stack(seq_embeddings, dim=0))
        y_list.append(int(label_value))

    if not x_list:
        raise ValueError("No valid samples were produced for sentiment evaluation.")

    X = torch.stack(x_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


class SentimentLSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_num_layers, output_size, bidirectional=False):
        super().__init__()
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc1 = nn.Linear(lstm_hidden_dim, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()
        self.fc_out = nn.Linear(128, output_size)

    def forward(self, x):
        _, (hidden_states, _) = self.lstm(x)
        final_hidden = hidden_states[-1]
        h = self.act1(self.fc1(final_hidden))
        h = self.act2(self.fc2(h))
        logits = self.fc_out(h)
        return logits


def save_training_metadata(metadata_path, metadata):
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def evaluate_batches(model, features, labels, batch_size, device, loss_fn):
    eval_ds = TensorDataset(features, labels)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    model.eval()
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train an LSTM sentiment classifier with an MLP head")
    parser.add_argument("--input_file_prefix", type=str, default="sent_d50", help="Prefix for input files")
    parser.add_argument("--limit", type=int, default=None, help="Limit sample count after dataset preparation (applied separately to train and test)")
    parser.add_argument("--lstm_hidden_dim", type=int, default=256, help="Number of hidden units in the LSTM")
    parser.add_argument("--lstm_num_layers", type=int, default=1, help="Number of stacked LSTM layers")
    parser.add_argument("--lstm_bidirectional", type=bool, default=False, help="Whether to use bidirectional LSTM")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save", action="store_true", help="Whether to save the model checkpoint")
    parser.add_argument("--eval", action="store_true", help="Whether to evaluate the model on test set")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "output_results" / "E_sent" / "ANN"), help="Output directory for checkpoint and metadata")
    parser.add_argument("--output_file_prefix", type=str, default="", help="Prefix for output files")
    args = parser.parse_args()

    sent_train_data, embedding_dim, emb_normalization_mode = ReadSENTInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_train.pkl", limit=args.limit)
    sent_test_data, _, _ = ReadSENTInputFile(CAST_INPUT_DIR / f"{args.input_file_prefix}_test.pkl", limit=args.limit)

    X_train, y_train = build_sentiment_samples(sent_train_data, embedding_dim)
    X_test, y_test = build_sentiment_samples(sent_test_data, embedding_dim)

    if X_train.ndim != 3 or X_train.shape[2] != embedding_dim:
        raise ValueError(f"Unexpected training tensor shape for temporal input: {tuple(X_train.shape)}; expected [N, T, {embedding_dim}]")

    if args.limit is not None:
        train_limit = min(args.limit, X_train.shape[0])
        test_limit = min(args.limit, X_test.shape[0])
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]
        X_test = X_test[:test_limit]
        y_test = y_test[:test_limit]

    sequence_length = X_train.shape[1]
    num_labels = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentimentLSTM(
        input_dim=embedding_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        bidirectional=args.lstm_bidirectional,
        output_size=num_labels,
    )
    model = model.to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir = Path(args.output_dir) or PROJECT_ROOT / "output_results" / "E_sent"
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_filename_base = "_".join([args.output_file_prefix or "sent_ann_lstm", now, f"e-{args.epochs}", f"bs-{args.batch_size}"])
    
    pprint(args.__dict__ | {
        "embedding_dim": embedding_dim,
        "embedding_normalization_mode": emb_normalization_mode,
        "sequence_length": sequence_length,
        "input_dim": embedding_dim,
    })

    metadata_file = output_dir / f"{run_filename_base}.json"
    training_metadata = {
        "training_config": {
            "training_start_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_end_date": None,
            "training_duration_s": None,
            "task": "sequence_binary_sentiment_lstm",
            "embedding_dim": int(embedding_dim),
            "embedding_normalization_mode": emb_normalization_mode,
            "sequence_length": int(sequence_length),
            "input_dim": int(embedding_dim),
            "lstm_hidden_dim": int(args.lstm_hidden_dim),
            "lstm_num_layers": int(args.lstm_num_layers),
            "layer_sizes": [int(args.lstm_hidden_dim), 256, 128, 2],
            "num_labels": num_labels,
            "num_training_samples": int(X_train.shape[0]),
            "num_test_samples": int(X_test.shape[0]),
            "device": str(device),
            "train_pos": int((y_train == 1).sum().item()),
            "train_neg": int((y_train == 0).sum().item()),
            "test_pos": int((y_test == 1).sum().item()),
            "test_neg": int((y_test == 0).sum().item()),
        },
        "results": {
            "epoch_train_loss": [],
            "epoch_train_accuracy": [],
            "test_loss": None,
            "test_accuracy": None,
        },
    }
    save_training_metadata(metadata_file, training_metadata)

    epoch_losses = []
    epoch_accuracies = []
    training_start_perf = time.perf_counter()
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
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

        epoch_loss = running_loss / max(1, running_total)
        epoch_acc = running_correct / max(1, running_total)
        epoch_losses.append(float(epoch_loss))
        epoch_accuracies.append(float(epoch_acc))

        training_metadata["results"]["epoch_train_loss"] = epoch_losses
        training_metadata["results"]["epoch_train_accuracy"] = epoch_accuracies
        save_training_metadata(metadata_file, training_metadata)

        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - training_start_perf
        avg_epoch = elapsed / float(epoch + 1)
        remaining = max(0, args.epochs - (epoch + 1))
        eta_min = (avg_epoch * remaining) / 60.0
        print(f"Epoch {epoch+1}/{args.epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f} | epoch_s: {epoch_time:.2f} | eta_min: {eta_min:.2f}")

    training_end_time = datetime.now()
    training_metadata["training_config"]["training_end_date"] = training_end_time.strftime("%Y-%m-%d %H:%M:%S")
    training_metadata["training_config"]["training_duration_s"] = time.perf_counter() - training_start_perf

    if args.save:
        model_output_path = output_dir / f"{run_filename_base}.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_class": "SentimentLSTM",
            "model_config": {
                "input_dim": embedding_dim,
                "lstm_hidden_dim": int(args.lstm_hidden_dim),
                "lstm_num_layers": int(args.lstm_num_layers),
                "output_size": num_labels,
                "layer_sizes": [int(args.lstm_hidden_dim), 256, 128, 2],
            },
            "label_maps": {"label_to_idx": {"negative": 0, "positive": 1}, "idx_to_label": {0: "negative", 1: "positive"}},
            "metrics": {"epoch_train_loss": epoch_losses, "epoch_train_accuracy": epoch_accuracies},
            "cli_args": vars(args),
        }
        torch.save(checkpoint, model_output_path)
        print(f"Model checkpoint saved to {model_output_path}")

    if args.eval:
        test_loss, test_acc = evaluate_batches(model, X_test, y_test, args.batch_size, device, loss_fn)
        training_metadata["results"]["test_loss"] = float(test_loss)
        training_metadata["results"]["test_accuracy"] = float(test_acc)
        print(f"Evaluation | samples={int(X_test.shape[0])} | loss={test_loss:.4f} | acc={test_acc:.4f}")

    save_training_metadata(metadata_file, training_metadata)
    print(f"Training metadata exported to {metadata_file}")


if __name__ == "__main__":
    main()
